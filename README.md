# SVG-metrics

This repository contains code to test one of our fine-tuned vision-language models for evaluating text-to-SVG models. We will release weights for all models, training scripts, and all datasets upon acceptance.

## Model

This code evaluates caption similarity using our fine-tuned VLM, by scoring SVG images on a scale of 1-5 based on:
- Main objects and their presence
- Object attributes (shape, size, color)
- Spatial relations and layout
- Counts and numbers
- Overall semantic match

Our model shows state-of-the-art correlation with human preference.

## Manual Setup

### 1. Install Dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

You may also need to install Cairo for SVG rendering:

```bash
sudo apt-get install libcairo2-dev
```


### 2. Configure HuggingFace Token

Our model is hosted on HuggingFace and currently requires authentication. Create a `.env` file in this directory with your HuggingFace token:

```bash
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

**Where to find your token:**
The HuggingFace token was provided at the end of the supplementary materials. 

## Usage

The `judge.py` script offers two modes:

### Mode 1: Evaluate from JSON File

Evaluate multiple SVG images from a JSON file.
We included a few examples from the test-set that we labelled in our work, containing a mix of ground truth SVGs from existing datasets and generated SVGs from text-to-SVG models.

```bash
python judge.py --json test.json
```

**JSON Format:**
The JSON file should contain a list of objects with `caption` and `svg_code` fields:
```json
[
  {
    "caption": "a black and white illustration of a stack of data",
    "svg_code": "<svg>...</svg>"
  },
  ...
]
```

### Mode 2: Evaluate Single SVG

Evaluate a single SVG file with a custom caption:

```bash
python judge.py --svg path/to/image.svg --caption "A red circle with blue border"
```

## Output

PNG files showing the rendered SVG with the caption, score, and reasoning
   - Located in the `results/` folder (or custom output directory)
   - Named `result_000.png`, `result_001.png`, etc. for JSON mode
   - Named `result_<svg_name>.png` for single SVG mode
