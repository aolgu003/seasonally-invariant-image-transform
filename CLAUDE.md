# CLAUDE.md

## Project Overview

This repository contains the official training code for the **Science Robotics** paper: *"A seasonally invariant deep transform for visual terrain-relative navigation"* (Fragoso et al., 2021). It trains deep neural networks (U-Net) to produce image transformations that are invariant to seasonal appearance changes (e.g., summer vs. winter), enabling robust visual terrain-relative navigation for robots.

Two training approaches are implemented:
- **NCC-based** (`siamese-ncc.py`): Optimizes for normalized cross-correlation registration
- **SIFT-based** (`siamese-sift.py`): Optimizes for SIFT feature matching (detector + descriptor losses)

## Repository Structure

```
├── siamese-ncc.py              # NCC-optimized training entry point
├── siamese-sift.py             # SIFT-optimized training entry point
├── siamese-inference.py        # Inference entry point
├── createTiledDataset.py       # Preprocessing: crops large images into tiles
├── requirements.txt            # Python dependencies (pip)
├── model/
│   ├── unet.py                 # U-Net architecture (1-ch in, 1-ch out, sigmoid output)
│   ├── unet_parts.py           # U-Net building blocks (DoubleConv, Down, Up, OutConv)
│   ├── correlator.py           # Normalized cross-correlation layer
│   ├── kornia_dog.py           # Difference of Gaussians multi-scale pyramid
│   └── kornia_sift.py          # SIFT descriptor extraction via Kornia
├── dataset/
│   ├── neg_dataset.py          # Siamese dataset for NCC training (with negative sampling)
│   ├── neg_sift_dataset.py     # Siamese dataset for SIFT training (with augmentation)
│   └── inference_dataset.py    # Dataset for inference
├── utils/
│   └── helper.py               # Normer, tensorboard logging, pyramid loss, inference helpers
├── data/
│   ├── coregistered_images/    # Raw paired images (on/ and off/ season subdirs)
│   └── samples/                # Sample images for inference demos
├── weights2try/                # Pre-trained model weights
│   ├── ctCo300dx1-weights/     # best_test_weights.pt, best_train_weights.pt
│   └── mtCo150ax1-weights/     # best_test_weights.pt, best_train_weights.pt
├── experiments/                # Output directory for experiment results and weights
└── runs/                       # TensorBoard log directory (gitignored)
```

## Tech Stack

- **Language**: Python 3
- **Deep Learning**: PyTorch 1.11.0, PyTorch Lightning 1.7.7
- **Computer Vision**: Kornia 0.6.6, OpenCV 4.5.5.64
- **Data Augmentation**: Albumentations 1.1.0
- **Image I/O**: Pillow, imageio, rasterio
- **Visualization**: TensorBoard, matplotlib
- **Scientific Computing**: NumPy, SciPy, scikit-image

## Environment Setup

```bash
# Create a Python 3 environment (conda recommended)
pip3 install -r requirements.txt
```

The `requirements.txt` contains pinned versions for all 207+ dependencies. Some packages are installed from local conda builds and git repos (e.g., detectron2, multipoint).

## Common Commands

### Data Preprocessing

Crop large orthorectified images into training tiles:

```bash
python createTiledDataset.py \
  --raw_data_dir=data/coregistered_images/off \
  --save_data_dir=data/training_pairs/off \
  --overlap_ratio=0 \
  --crop_width=600 \
  --crop_height=600

python createTiledDataset.py \
  --raw_data_dir=data/coregistered_images/on \
  --save_data_dir=data/training_pairs/on \
  --overlap_ratio=0 \
  --crop_width=600 \
  --crop_height=600
```

### Training (NCC)

```bash
python siamese-ncc.py \
  --exp_name=correlation-toy-example \
  --training_data_dir=data/training_pairs/ \
  --validation_data_dir=data/training_pairs/ \
  --batch-size=4 \
  --epochs=100 \
  --device=0 \
  --num_workers=4
```

### Training (SIFT)

```bash
python siamese-sift.py \
  --exp_name=sift-toy-example \
  --training_data_dir=data/training_pairs/ \
  --validation_data_dir=data/training_pairs/ \
  --subsamples=100 \
  --crop_width=64 \
  --batch-size=2 \
  --zeta=10 \
  --gamma=1 \
  --epochs=100
```

Key SIFT-specific args:
- `--zeta`: Descriptor loss weight (default: 10)
- `--gamma`: Detector loss weight (default: 1)
- `--subsamples`: Number of random crop subsamples per batch (default: 100)
- `--crop_width`: Crop size for pyramid loss patches (default: 64)

### Inference

```bash
python siamese-inference.py \
  --data_dir=data/samples/fakeplaceid_fakequad_000015_on.png \
  --output_dir=correlation-toy-example/sample-outputs \
  --weights_path=correlation-toy-example/weights/best_test_weights.pt \
  --mean=0.4 \
  --std=0.12
```

### TensorBoard

```bash
tensorboard --logdir=runs/
```

## Architecture Details

### Model

- **U-Net** with 1 grayscale input channel and 1 output channel
- Encoder: 5 levels (64 -> 128 -> 256 -> 512 -> 512 channels)
- Decoder: 4 upsampling levels with skip connections
- Bilinear upsampling (default), transpose convolution available
- Sigmoid activation on output (produces [0, 1] range images)
- Based on [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

### Training Pipeline

1. **Data**: Paired coregistered "on-season" and "off-season" images, tiled into patches
2. **Dataset**: Siamese pairs with 50% negative (mismatched) samples
3. **Loss (NCC)**: MSE between predicted NCC score and label (1=match, 0=mismatch)
4. **Loss (SIFT)**: Weighted combination of DoG pyramid correlation loss and SIFT descriptor distance loss: `gamma * sift_loss + zeta * pyramid_loss`
5. **Optimizer**: Adam (lr=1e-5) with ExponentialLR scheduler (gamma=0.995 for NCC, 0.99 for SIFT)
6. **Outputs**: Best test/train weights saved to `experiments/{exp_name}/weights/`

### Data Format

- **Input**: PNG images (grayscale or RGB converted to grayscale)
- **Directory structure**: Paired images in `on/` and `off/` subdirectories with matching filenames
- **Normalization stats**:
  - NCC dataset: on-season mean=0.49, std=0.12; off-season mean=0.44, std=0.10
  - SIFT dataset: on-season mean=0.49, std=0.135; off-season mean=0.44, std=0.12
  - Inference defaults: mean=0.4, std=0.12

## Key Conventions

- Random seeds are fixed for reproducibility (torch.manual_seed(0), random.seed(0), np.random.seed(0))
- `torch.autograd.set_detect_anomaly(True)` is enabled in training scripts for debugging
- `cudnn.deterministic = True` and `cudnn.benchmark = False` for reproducibility
- Weights are saved as PyTorch state dicts (`.pt` files)
- Experiment outputs go to `experiments/{exp_name}/` and TensorBoard logs to `runs/{exp_name}/`
- Mixed precision training via NVIDIA Apex is supported but optional (auto-detected)

## Testing

There is no formal test suite. Validation is performed during training via the validation dataset split. Verify model quality by:

1. Monitoring TensorBoard loss curves during training
2. Running inference and visually inspecting transformed images
3. Comparing NCC scores or SIFT feature matches between transformed on/off season pairs

## Code Style

- No linter or formatter configuration files are present (black is installed but unconfigured)
- No pre-commit hooks
- No CI/CD pipelines
- Commit messages use imperative mood, lowercase, brief descriptions

## Notes for AI Assistants

- This is research code accompanying a published paper. Avoid refactoring the core training logic unless explicitly requested.
- The SIFT-based training is more sensitive to hyperparameters than NCC. The README warns about tuning `--zeta` and `--gamma`.
- The `data/training_pairs/` directory is gitignored; it must be generated from `data/coregistered_images/` via `createTiledDataset.py`.
- Pre-trained weights are available in `weights2try/` for quick inference experiments.
- GPU is assumed for training (`--device=0`); CPU fallback exists but will be slow.
