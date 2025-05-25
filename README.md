# Demucs ONNX Denoiser

A PyTorch-based speech denoising project built on top of Facebook Research's denoiser repository. This repository provides training pipelines, model definitions, and ONNX export scripts for a Demucs-inspired denoiser that you can integrate into your applications via ONNX modules.
The original project can be found at https://github.com/facebookresearch/denoiser.

## Features

- Two-step training pipeline:

  1. Valentini dataset pretraining

  2. MS-SNSD fine-tuning

- Custom augmentation, loss functions, and model definitions

- Exports encoder and decoder to ONNX for cross-platform inference

- Simple Python application (app.py) for real-time denoising

# Getting Started

Prerequisites

- Python 3.8 or higher

- pip for managing Python packages

- Datasets:

  1. [Valentini dataset](https://datashare.ed.ac.uk/handle/10283/2791) (Edinburgh DataShare)

  3. [Microsoft MS-SNSD repository](https://github.com/microsoft/MS-SNSD)
 
# Installation

Clone this repository:
```
git clone https://github.com/GitStroberi/demucs-onnx.git
cd demucs-onnx
```
Create and activate a virtual environment:

```
conda env create -f environment.yml -n demucs-onnx
conda activate demucs-onnx
```

Dataset Preparation

1. Download the Valentini and MS-SNSD datasets.

2. Unzip and organize them into folders on your local machine.

3. Update the file paths in both training scripts (Denoiser-Valentini.py and Denoiser-MS-SNSD.py) to point to your dataset directories.

# Training (optional)

1. Valentini dataset training

Run the first-stage training on the Valentini dataset:

```
python Denoiser-Valentini.py
```

This script will:

- Load and preprocess Valentini noisy and clean pairs

- Apply augmentation (via augmentation.py)

- Train the Demucs causal model (defined in model_def.py)

- Save checkpoints

2. MS-SNSD Fine-tuning

After completing Valentini training, fine-tune on MS-SNSD:

```
python Denoiser-MS-SNSD.py
```

This script will:

- Load the pretrained weights from the Valentini stage

- Continue training on MS-SNSD data for better generalization

- Save final model checkpoint (sample model checkpoint present in the repository as demucs_model_finetune.pth)

# Inference & ONNX Export

Before running the demo app, you must export the encoder and decoder:

```
python inference.py
```

This script will:

- Split the trained model into three parts: encoder, LSTM bottleneck, and decoder

- Export the encoder and decoder to ONNX format (.onnx files)

- Start a simple real time inference streaming loop using the model

# Running the App

Once you have your encoder.onnx and decoder.onnx files, run the demo application:

```
python app.py
```


# License

This project incorporates code from facebookresearch/denoiser, which is licensed under CC BY-NC 4.0 (see LICENSE).
