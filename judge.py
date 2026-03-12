#!/usr/bin/env python3
"""
SVG Caption Similarity Judge

This script evaluates how well SVG images match their text descriptions using
a fine-tuned vision-language model from HuggingFace.

Usage:
    # Evaluate SVGs from a JSON file
    python judge.py --json test.json

    # Evaluate a single SVG file with a caption
    python judge.py --svg path/to/image.svg --caption "A red circle"
"""

import json
import argparse
import os
import re
from typing import Optional, Tuple, List
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import AutoProcessor, AutoModelForImageTextToText
import cairosvg
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN not found in environment. Set it in .env file.")
    print("You may encounter errors when loading the model from HuggingFace.")


# ─── SVG Rendering ────────────────────────────────────────────────────────────

def svg_to_png(svg_string: str) -> Image.Image:
    """Convert SVG string to PNG image."""
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:svg|xml)?\s*", "", svg_string.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())

    # Strip leading garbage before <svg
    svg_start = cleaned.find("<svg")
    if svg_start > 0:
        cleaned = cleaned[svg_start:]

    # Extract only the first complete <svg>...</svg> block
    svg_match = re.search(r"<svg[\s\S]*</svg>", cleaned, re.IGNORECASE)
    if svg_match:
        cleaned = svg_match.group()
    else:
        # No closing tag found — truncated SVG, try appending it
        cleaned = cleaned + "</svg>"

    try:
        png_bytes = cairosvg.svg2png(bytestring=cleaned.encode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to render SVG: {e}")

    # Open with alpha channel
    image = Image.open(BytesIO(png_bytes)).convert("RGBA")

    # Create white background
    white_bg = Image.new("RGB", image.size, (255, 255, 255))

    # Composite SVG on white background
    white_bg.paste(image, mask=image.split()[3])  # Use alpha channel as mask

    return white_bg


# ─── Prompt ───────────────────────────────────────────────────────────────────

def get_caption_prompt_text(caption: str) -> str:
    """Generate evaluation prompt for caption similarity."""
    criteria = (
        "- Main objects and their presence\n"
        "- Object attributes (shape, size, color)\n"
        "- Spatial relations and layout\n"
        "- Counts and numbers\n"
        "- Overall semantic match"
    )

    rubric = (
        "5: Very strong match; main objects, layout, and key attributes align\n"
        "4: Good match; overall scene corresponds with only minor issues\n"
        "3: Partial match; several core elements align but some details wrong\n"
        "2: Weak match; similar topic but multiple important errors\n"
        "1: Unrelated or minimal overlap"
    )

    format_block = (
        "Format your response as:\n"
        "<think>Your brief reasoning here</think>\n"
        "<score>X</score>\n\n"
        "where X is a number from 1 to 5."
    )

    return (
        f"You are a concise evaluator of text-to-SVG faithfulness. Judge how\n"
        f"well a generated SVG image matches its textual description.\n\n"
        f"**Task**: Compare the generated image to the text description below.\n\n"
        f"**Text Description**: {caption}\n\n"
        f"**Evaluation Criteria**:\n{criteria}\n\n"
        f"**Scoring Rubric (1-5)**:\n{rubric}\n\n"
        f"**Instructions**:\n"
        f"First, briefly analyze what you observe in the image and how it compares to the description.\n"
        f"Then, provide your final score.\n\n"
        f"{format_block}"
    )


def build_messages(caption: str, image: Image.Image) -> List[dict]:
    """Build messages for the model."""
    prompt_text = get_caption_prompt_text(caption)
    system_message = (
        "You are a vision-language evaluator that thinks step-by-step. "
        "Always analyze the image first, then provide your predictions in the exact format requested."
    )
    return [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_name: str = "anon-submission-data/qwen3-vl-8b-thinking-grpo-text-2-svg-judge"):
    """Load the caption similarity model from HuggingFace."""
    print("Loading model...")
    print("This may take a few minutes on first run...")

    processor_name = "Qwen/Qwen3-VL-8B-Thinking"

    processor = AutoProcessor.from_pretrained(
        processor_name,
        trust_remote_code=True,
        token=HF_TOKEN
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    model.eval()
    print("Model loaded successfully!")
    return model, processor


# ─── Inference ────────────────────────────────────────────────────────────────

def extract_score_from_response(response_text: str) -> Tuple[Optional[int], str]:
    """Extract score and thinking from model response."""
    thinking_content = ""
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL | re.IGNORECASE)
    if think_match:
        thinking_content = think_match.group(1).strip()

    score_match = re.search(r'<score>\s*(\d)\s*</score>', response_text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        if 1 <= score <= 5:
            return score, thinking_content

    # Fallback: look for digits after thinking block
    if think_match:
        after_think = response_text[think_match.end():]
        digits = re.findall(r'\b([1-5])\b', after_think)
        if digits:
            return int(digits[0]), thinking_content

    # Last resort: find any valid score
    digits = re.findall(r'\b([1-5])\b', response_text)
    if digits:
        return int(digits[-1]), thinking_content

    return None, thinking_content


def evaluate_svg(
    model,
    processor,
    image: Image.Image,
    caption: str,
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> Tuple[int, str, str]:
    """Evaluate a single SVG image against its caption."""
    messages = build_messages(caption, image)

    # Build prompt using chat template
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # Process inputs
    model_inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    model_inputs = {
        k: (v.to(model.device) if torch.is_tensor(v) else v)
        for k, v in model_inputs.items()
    }

    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
        )

    # Decode output
    input_len = model_inputs["input_ids"].shape[1]
    output_text = processor.tokenizer.decode(
        output_ids[0, input_len:],
        skip_special_tokens=True,
    )

    # Extract score and thinking
    score, thinking = extract_score_from_response(output_text)
    if score is None:
        print(f"WARNING: Could not parse score from response. Using default score of 3.")
        print(f"Response: {output_text[:200]}...")
        score = 3

    return score, thinking, output_text


# ─── Visualization ────────────────────────────────────────────────────────────

def save_result_figure(
    image: Image.Image,
    caption: str,
    score: int,
    thinking: str,
    output_path: str,
):
    """Save a figure showing the rendered SVG, caption, and score."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Display image
    ax.imshow(image)
    ax.axis('off')

    # Add title with score
    score_colors = {1: 'red', 2: 'orange', 3: 'yellow', 4: 'lightgreen', 5: 'green'}
    title = f"Score: {score}/5"
    ax.set_title(title, fontsize=16, fontweight='bold',
                 color=score_colors.get(score, 'black'), pad=20)

    # Add caption and thinking as text below the image
    caption_text = f"Caption: {caption}"
    thinking_text = f"Reasoning: {thinking[:200]}..." if len(thinking) > 200 else f"Reasoning: {thinking}"

    fig.text(0.5, 0.02, caption_text, ha='center', fontsize=10,
             wrap=True, style='italic')
    fig.text(0.5, 0.07, thinking_text, ha='center', fontsize=9,
             wrap=True, color='gray')

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved result to: {output_path}")


# ─── Main Functions ───────────────────────────────────────────────────────────

def evaluate_from_json(json_path: str, output_dir: str = "results"):
    """Evaluate all SVGs in a JSON file."""
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {json_path}")

    # Load model
    model, processor = load_model()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate each item
    results = []
    for i, item in enumerate(data):
        caption = item.get('caption', '')
        svg_code = item.get('svg_code', '')

        if not caption or not svg_code:
            print(f"Skipping item {i}: missing caption or svg_code")
            continue

        print(f"\n[{i+1}/{len(data)}] Evaluating: {caption[:60]}...")

        try:
            # Render SVG
            image = svg_to_png(svg_code)

            # Evaluate
            score, thinking, full_output = evaluate_svg(model, processor, image, caption)

            # Save result
            output_path = os.path.join(output_dir, f"result_{i:03d}.png")
            save_result_figure(image, caption, score, thinking, output_path)

            # Store result
            results.append({
                'index': i,
                'caption': caption,
                'score': score,
                'thinking': thinking,
            })

            print(f"  Score: {score}/5")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Print summary
    if results:
        avg_score = sum(r['score'] for r in results) / len(results)
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Evaluated: {len(results)} items")
        print(f"  Average score: {avg_score:.2f}/5")
        print(f"  Results saved to: {output_dir}/")
        print(f"{'='*60}")

        # Save summary JSON
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump({
                'total_items': len(results),
                'average_score': avg_score,
                'results': results,
            }, f, indent=2)
        print(f"Summary saved to: {summary_path}")
    else:
        print("No results to summarize.")


def evaluate_single_svg(svg_path: str, caption: str, output_dir: str = "results"):
    """Evaluate a single SVG file with a given caption."""
    print(f"Loading SVG from: {svg_path}")
    print(f"Caption: {caption}")

    # Load SVG
    with open(svg_path, 'r') as f:
        svg_code = f.read()

    # Load model
    model, processor = load_model()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Render SVG
        image = svg_to_png(svg_code)

        # Evaluate
        score, thinking, full_output = evaluate_svg(model, processor, image, caption)

        # Save result
        svg_name = Path(svg_path).stem
        output_path = os.path.join(output_dir, f"result_{svg_name}.png")
        save_result_figure(image, caption, score, thinking, output_path)

        # Print result
        print(f"\n{'='*60}")
        print(f"Result:")
        print(f"  Score: {score}/5")
        print(f"  Reasoning: {thinking}")
        print(f"  Figure saved to: {output_path}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"ERROR: {e}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SVG images against text captions using a fine-tuned VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate SVGs from a JSON file
  python judge.py --json test.json

  # Evaluate a single SVG with a caption
  python judge.py --svg path/to/image.svg --caption "A red circle with blue border"
        """
    )

    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Path to JSON file containing SVG data (default: test.json if neither --json nor --svg is specified)'
    )

    parser.add_argument(
        '--svg',
        type=str,
        help='Path to a single SVG file'
    )

    parser.add_argument(
        '--caption',
        type=str,
        help='Caption for the SVG (required when using --svg)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )

    args = parser.parse_args()

    # Determine mode
    if args.svg:
        # Single SVG mode
        if not args.caption:
            parser.error("--caption is required when using --svg")
        evaluate_single_svg(args.svg, args.caption, args.output)
    else:
        # JSON mode
        json_path = args.json if args.json else 'test.json'
        evaluate_from_json(json_path, args.output)


if __name__ == '__main__':
    main()
