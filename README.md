# Tarot-SAM3 Reproduction

This repository is a clean reproduction workspace for **Tarot-SAM3: Training-free SAM3 for Any Referring Expression Segmentation**.

The paper proposes a training-free RES pipeline that combines:

- **Qwen2.5-VL** as the multimodal reasoning model.
- **SAM3** as the text / box / point prompted mask generator.
- **DINOv3** as the dense visual feature extractor for mask self-refinement.

No task-specific model training is required. The main work is to reproduce the inference pipeline, prepare RES datasets, and evaluate gIoU / cIoU consistently with the paper.

## Project Status

Initial scaffold:

- [x] Directory structure
- [x] Environment files
- [x] Configuration templates
- [x] Prompt templates
- [ ] Model download
- [ ] Dataset download
- [ ] SAM3 / Qwen / DINO wrappers
- [ ] ERI pipeline
- [ ] MSR pipeline
- [ ] Evaluation scripts

## Directory Layout

```text
Tarot-SAM3/
├── configs/                  # YAML configs for datasets and inference
├── checkpoints/              # Local model weights
├── datasets/                 # RefCOCO / ReasonSeg data
├── external/                 # Third-party repositories
├── tarot_sam3/               # Main Python package
├── prompts/                  # MLLM prompt templates
├── scripts/                  # Setup, preparation, inference, evaluation scripts
├── notebooks/                # Debug notebooks
├── outputs/                  # Predictions, logs, tables, visualizations
└── tests/                    # Unit tests
```

## Environment

Recommended platform:

- Linux server or WSL2
- Python 3.12
- CUDA 12.6+
- PyTorch 2.10.0 CUDA 12.8 wheel
- 1 GPU for debugging; 4 x A800 80G were used in the paper

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate tarot-sam3
```

Or install with pip:

```bash
conda create -n tarot-sam3 python=3.12 -y
conda activate tarot-sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## External Repositories

Install SAM3:

```bash
cd external
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install -e ".[notebooks]"
```

Install RefCOCO API:

```bash
cd external
git clone https://github.com/lichengunc/refer.git
cd refer
make
```

## Model Checkpoints

Log in to Hugging Face first:

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
hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir checkpoints/dinov3-vitb16
```

SAM3 checkpoints are managed by the official SAM3 package and Hugging Face access flow. Put any manually downloaded weights under:

```text
checkpoints/sam3/
```

## Datasets

Expected dataset layout:

```text
datasets/
├── refer_seg/
│   ├── images/mscoco/images/train2014/
│   ├── refcoco/
│   ├── refcoco+/
│   └── refcocog/
└── reason_seg/
    └── ReasonSeg/
        ├── train/
        ├── val/
        ├── test/
        └── explanatory/
```

Paper evaluation:

- Explicit RES: RefCOCO, RefCOCO+, RefCOCOg
- Implicit RES: ReasonSeg from LISA
- Explicit metric: gIoU
- Implicit metrics: gIoU and cIoU

## Method Milestones

1. Run individual demos for SAM3, Qwen2.5-VL, and DINOv3.
2. Implement dataset readers and metric checks.
3. Implement ERI:
   - Reasoning-assisted prompt options
   - Target name extraction / inference
   - Text prompt augmentation
   - BBox prompt augmentation
   - Prompt-consistency filtering
   - Intra-prompt mask selection
4. Implement DINO point prompt generation.
5. Implement MSR:
   - Inter-prompt mask selection
   - Discriminative region extraction
   - Region affiliation judgment
   - Object-aware point prompt modification
6. Run ablations and compare against the paper.

## Key Paper Settings

```yaml
mllm: Qwen2.5-VL-7B-Instruct
dino: DINOv3 ViT-B
tau: 0.80
sneg: 0.30
```

## First Smoke Test

After implementing the wrappers, the first command should be a tiny single-sample run:

```bash
python scripts/run_inference.py \
  --config configs/default.yaml \
  --image path/to/image.jpg \
  --query "the man with blue backpack" \
  --output outputs/visualizations/smoke_test.png
```

Then evaluate a small subset:

```bash
python scripts/eval_refcoco.py \
  --config configs/refcoco.yaml \
  --split testA \
  --max-samples 50
```

