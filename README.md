### Prompted Segmentation for Drywall & Crack Quality Assurance

Author: Soumen
Institute: IIT Bhubaneswar
Role Applied: AI Research Apprentice @ 10xConstruction
Date: November 2025

## Problem Motivation

Crack detection and drywall quality inspection are critical but manual tasks in construction QA.
This project explores whether a prompt-based segmentation model can reliably identify:

Structural cracks

Drywall taping regions

using natural language instructions, enabling scalable inspection for autonomous robots and vision systems.

## Project Overview

This project fine-tunes CLIPSeg, a text-conditioned image segmentation model, to segment construction defects based on prompts such as:

segment crack

segment drywall taping area

Instead of training a separate model per defect type, the system learns to generalize using text prompts, making it flexible for real-world inspection scenarios.

## Model Used

CLIPSeg (Hugging Face Transformers)

Vision encoder + text encoder

Prompt-guided pixel-wise segmentation

Why CLIPSeg?

Supports open-vocabulary segmentation

Suitable for environments where defect types evolve

Aligns well with foundation-model-based inspection pipelines

## Qualitative Results (Model Output)

Below is a grid visualization showing input images and predicted segmentation masks for cracks across diverse wall surfaces:

Concrete

Brick

Painted drywall

Rough plaster

## Crack Segmentation Results

Observation:

The model captures thin, elongated crack structures

Robust across texture and illumination variations

Handles both horizontal and vertical cracks reasonably well

## Training & Evaluation
Metrics Used

Dice Coefficient

Mean IoU (mIoU)

Training Objective

Learn text–image alignment for defect localization

Reduce dependence on rigid class labels

## Setup Instructions
## Environment

Tested on:

Python 3.10+

PyTorch 2.x

CUDA 11.8

Hugging Face Transformers ≥ 4.44

scikit-learn, matplotlib, Pillow, tqdm

## Install dependencies:

pip install torch torchvision torchaudio transformers scikit-learn matplotlib pillow tqdm

## Train the Model
python train_clipseg_from_txt.py

## Test & Visualize Results
python test_and_visualize.py


Segmentation outputs and metrics are saved in:

results/

## Applications

Construction site QA automation

Crack monitoring over time

Vision-enabled inspection robots

Foundation models for infrastructure health assessment

## Future Work

Multi-prompt learning (crack + spalling + moisture)

Temporal crack progression tracking

Integration with depth or thermal data

Comparison with SAM / MedSAM-style prompting

## Key Takeaway

This project demonstrates that prompt-driven segmentation can be a viable alternative to rigid class-based models for construction inspection — reducing retraining costs while improving flexibility.
