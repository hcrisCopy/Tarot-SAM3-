# Tarot-SAM3 Single-Image Reproduction

This repository reproduces the single-image inference path of **Tarot-SAM3:
Training-free SAM3 for Any Referring Expression Segmentation**.

The implementation is training-free. It freezes three pretrained backbones and
uses prompt engineering plus mask self-refinement:

- **Qwen2.5-VL**: multimodal reasoning, JSON parsing, mask/region judgment.
- **SAM3**: text, box, and point prompted mask generation.
- **DINOv3**: dense feature similarity for point prompt generation and mask refinement.

The current code focuses on **one image and one referring expression**. Dataset
evaluation scripts are still placeholders for future RefCOCO/ReasonSeg work.

## Method Summary

Tarot-SAM3 has two phases.

### 1. ERI: Expression Reasoning Interpreter

ERI turns a raw expression into several SAM3-compatible prompts:

1. Parse whether the query is explicit or implicit.
2. Extract or infer the target name.
3. Extract refer objects and visual attributes.
4. Generate three target text prompts.
5. Use SAM3 text prompts to produce text-mask candidates.
6. Rephrase the query into short and long target-centric expressions.
7. Ask Qwen2.5-VL for target bounding boxes.
8. Use SAM3 box prompts to produce box-mask candidates.
9. Filter text masks by box consistency with `tau`.
10. Select the best text mask and best box mask.
11. Use their high-confidence overlap and DINOv3 similarity to create positive
    and negative point prompts.
12. Use SAM3 point prompts to produce a point-mask candidate.

### 2. MSR: Mask Self-Refining

MSR selects and optionally corrects the ERI masks:

1. Select the best mask across text, box, and point prompt outputs.
2. Compare the selected mask with the point-prompt mask.
3. Partition discriminative regions:
   - `best_only = selected_best - point_mask`
   - `point_only = point_mask - selected_best`
4. Ask Qwen2.5-VL whether each region belongs to the same target.
5. If `best_only` is not target, treat it as over-segmentation and move the
   negative point there.
6. If `point_only` is target, treat it as under-segmentation and add a positive
   point there.
7. Re-query SAM3 with updated point prompts for the final mask.

Default paper hyperparameters:

```yaml
tau: 0.80
sneg: 0.30
```

## Directory Layout

```text
configs/                  YAML configs
checkpoints/              Local model weights
external/                 Third-party repositories, including SAM3
prompts/                  Qwen2.5-VL prompt templates
scripts/                  CLI entrypoints
tarot_sam3/
  config.py               YAML config loader
  models/                 Qwen, SAM3, and DINO wrappers
  eri/                    Expression Reasoning Interpreter
  msr/                    Mask Self-Refining
  pipeline/               Single-image pipeline
  utils/                  JSON, geometry, prompt, and visualization helpers
outputs/                  Visualizations and intermediate JSON
tests/                    Unit tests
```

## Environment

Recommended platform:

- Linux server or WSL2
- Python 3.12
- CUDA 12.6+
- PyTorch 2.10.0 CUDA 12.8 wheel

Create the environment:

```bash
conda create -n tarot-sam3 python=3.12 -y
conda activate tarot-sam3

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install Cython
```

Install SAM3:

```bash
cd external
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install -e ".[notebooks]"
```

## Checkpoints

Login to Hugging Face:

```bash
hf auth login
```

Download Qwen2.5-VL:

```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir checkpoints/Qwen2.5-VL-7B-Instruct
```

Download DINOv3:

```bash
hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir checkpoints/dinov3-vitb16
```

Download SAM3 after access is approved:

```bash
hf download facebook/sam3 --local-dir checkpoints/sam3
```

Expected SAM3 checkpoint:

```text
checkpoints/sam3/sam3.pt
```

## Run Single-Image Inference

```bash
python scripts/run_inference.py \
  --config configs/default.yaml \
  --image path/to/image.jpg \
  --query "the man with blue backpack" \
  --output outputs/visualizations/smoke_test.png
```

Outputs:

```text
outputs/visualizations/smoke_test.png      final mask overlay
outputs/visualizations/smoke_test.json     structured intermediate results
outputs/intermediates/*.png                candidate and region panels
```

The JSON file records:

- ERI reasoning output
- augmented text prompts
- rephrased expressions
- Qwen-predicted boxes
- selected text/box/point masks
- DINO-generated positive and negative points
- MSR over/under-segmentation decisions
- final mask metadata

## Prompt Templates

Prompt templates are in `prompts/`. They are intentionally separated so you can
tune behavior without changing code.

- `eri_parse_prompt.txt`: parses the expression into structured reasoning,
  including target name, refer objects, attributes, and possible SAM3 confusion.
- `target_augmentation_prompt.txt`: expands the target name into three short
  SAM3-compatible text prompts.
- `criterion_prompt.txt`: describes how refer objects should relate to the
  target, used as semantic context for rephrasing and judgment.
- `rephrase_prompt.txt`: converts the original query into short and long
  target-centric expressions for bbox grounding.
- `bbox_prompt.txt`: asks Qwen2.5-VL for one target bounding box in original
  image pixel coordinates.
- `mask_selection_prompt.txt`: selects the best candidate mask from a rendered
  panel of indexed masks.
- `region_affiliation_prompt.txt`: decides whether a red discriminative region
  belongs to the same object as the green high-confidence target region.

All prompts are expected to return strict JSON. The code has a tolerant JSON
extractor, but cleaner JSON responses make debugging much easier.

## Important Config Values

```yaml
models:
  mllm:
    local_path: checkpoints/Qwen2.5-VL-7B-Instruct
  sam3:
    checkpoint_dir: checkpoints/sam3
    confidence_threshold: 0.25
    dtype: float32
    allow_low_precision: false
  dino:
    local_path: checkpoints/dinov3-vitb16

method:
  tau: 0.80
  sneg: 0.30
  text_prompt_topk: 3
  mask_score_topk: 5
  point_prompt:
    anchor_count: 5
  msr:
    enable: true
    max_refine_rounds: 1
```

If SAM3 returns too few masks, lower `models.sam3.confidence_threshold`. If text
masks are filtered too aggressively, lower `method.tau`. Keep
`models.sam3.allow_low_precision` as `false` for the first successful single-image
run; SAM3 currently creates some internal float32 tensors, so fp16/bf16 can
trigger dtype mismatch errors in some environments.

## Code Status

Implemented:

- Single-image config loading
- Qwen2.5-VL JSON reasoning wrapper
- SAM3 text, box, and point prompt wrapper
- DINOv3 dense similarity wrapper
- ERI phase
- MSR phase
- Single-image CLI
- Intermediate JSON and visualization artifacts

Pending:

- RefCOCO dataset reader
- ReasonSeg dataset reader
- Full dataset evaluation loops
- Paper table reproduction and ablations
