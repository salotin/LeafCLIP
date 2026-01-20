# LeafCLIP: A Detection Method for the Lesion Areas of Outdoor Plants in Complex Lighting Scenarios

**LeafCLIP** is a robust lesion-area detection framework designed for **outdoor plant images under complex field conditions**, where severe illumination changes (shadows), occlusion, and background clutter cause strong domain shift.

This repository provides the official implementation of the **LeafCLIP** framework proposed in:

 

---

## 📌 Key Challenges

Outdoor plant lesion detection is difficult due to:
- **Irregular shadows & photometric degradation** 
- **Occlusions & complex backgrounds** 
- **Limited labeled lesion masks**

---

## ✅ Main Contributions

LeafCLIP contains two major components:

### 1) Cascaded CLIP for lesion localization
A multi-layer CLIP-based architecture that performs **multi-scale anomaly matching**, including:
- **Structure-preserving synthetic anomaly generation** to enrich training diversity
- **Multi-scale CLIP feature alignment** to improve small lesion detection under occlusion
- **Lightweight adapters** to fuse multi-depth CLIP visual tokens
- **Domain-specific learnable prompts (CoOp)** for better cross-modal alignment

### 2) Dynamic Mask-aware Diffusion Model (DMDM) for shadow removal
A diffusion-based shadow correction module that:
- reconstructs **illumination-invariant representations**
- jointly refines shadow masks and restores degraded regions
- improves robustness under outdoor complex illumination

---

## 🧱 Project Structure (Core Files)

> Note: the directory only displays part of the content.

```bash
LeafCLIP/
├── Cascaded CLIP
    ├── train.py                 # Cascaded CLIP main training entry
    ├── test.py               
    ├── config
        └── plant.yaml           # config for Cascaded CLIP training
    ├── datasets                 # Unsupervised Learning Data Construction
        └── dataset.py
    ├── models
        ├── Adapter.py           # lightweight adapters for multi-layer CLIP fusion
        ├── Necker.py            # multi-scale token alignment (upsampling)
        ├── CoOp.py              # learnable text prompts (PromptLearner + TextEncoder)
        └── MapMaker.py          # anomaly map generation via similarity
    ├── leafsyn
        ├── labelling.py
        ├── task_shape.py
        ├── tasks.py
        └── utils.py
    ├── open_clip                # CLIP Basic Framework
        ├── ***
        └── ***
    ├── utils                    # Loss function settings
        ├── misc_helper.py
        └── losses.py
    └── ***
└── DMDM
    ├── diffusionSDRM.py         # diffusion UNet backbone (DensePosteriorConditionalUNet)
    ├── DMDMtrain.py             # diffusion shadow removal training
    ├── DMDMtest.py              # diffusion test script
    ├── shadow_infer.py          # batch inference for shadow removal
    └── imresize.py              # resize utility
```

--- 
## 🔧 Environment Setup

Install a CUDA-compatible PyTorch version (example):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
--- 
## 📦 Datasets

Expected format (training set)

Your TrainDataset loads from:

```bash
{data_root}/{train_dataset}/
├── images/
└── samples/
    └── train.json
```
Each line in train.json is a JSON record containing filename (image name).

--- 
## 🏗️ Method Overview
LeafCLIP pipeline:

### Step A — DMDM (Shadow Removal, optional but recommended)

- **Use diffusion model to normalize illumination before lesion detection.**

 Input: shadowed leaf image
 Output: shadow-corrected image

### Step B — Cascaded CLIP for lesion anomaly localization

- **Uses OpenCLIP visual tokens from multiple depths and fuses them.**

Core steps in train.py:
Extract multi-layer CLIP tokens;
Align token spatial scales via Necker;
Channel project with Adapter;
Learn domain prompts with CoOp PromptLearner;
Generate anomaly map with MapMaker;
Optimize (PromptLearner + Adapter only, CLIP frozen);

--- 

## 🚀 Training

### Synthetic Anomaly Generation (Training)

LeafCLIP leverages structure-aware lesion synthesis for robust training.

In dataset.py, training anomalies are generated using MedSyn tasks.


### 1) Shadow Removal (Diffusion Module)

This module removes irregular outdoor shadows before training.

Train diffusion shadow removal model.
```bash
python DMDMtrain.py
```

The diffusion training uses Accelerate + fp16 and saves checkpoints to: ``` experiments/state_xxxxxx.bin```

Test diffusion model. 
```bash
python DMDMtest.py
```
Batch inference for shadow removal (recommended)
```bash
python shadow_infer.py \
  --ckpt experiments/state_xxxxxx.bin \
  --input_dir path/to/shadow_images \
  --output_dir path/to/deshadow_results
```
```shadow_infer.py``` supports optional mask input and applies dilation refinement before feature encoding.


### 2) Prepare config

Edit plant.yaml to configure:

* CLIP backbone name
* image size
* data paths
* layers to extract
* prompt settings

### 3) Run training
```bash
python train.py --config_path plant.yaml --k_shot XXX
```

During training, CLIP backbone weights are frozen, only adapters + prompts are optimized.

### 4) Run testing
To test the LeafCLIP on the dataset:
```bash
python  test.py --config_path config/plant.yaml  --checkpoint_path xxx.pkl
```





