Prompt-based Segmentation for Drywall Crack Detection

Author: Soumen
Affiliation: IIT Bhubaneswar

Overview

This project investigates the use of prompt-driven image segmentation for detecting cracks and drywall defects in construction images.
Instead of training a fixed-class segmentation network, we fine-tune CLIPSeg, a text-conditioned model, to localize defects using natural language prompts such as:

segment crack

segment drywall taping area

The motivation is to build a flexible inspection system suitable for robotic or automated construction quality assessment.

Method

Model: CLIPSeg (Hugging Face Transformers)

Input: RGB image + text prompt

Output: Binary segmentation mask corresponding to the prompt

Training: Supervised fine-tuning using crack masks aligned with text prompts

This approach reduces the need for rigid class definitions and allows extension to new defect types via prompt engineering.

Qualitative Results

The following figure shows representative segmentation results on different wall and surface textures.
White regions correspond to the predicted crack masks.

Evaluation

Performance is evaluated using standard segmentation metrics:

Dice coefficient

Mean IoU

Both quantitative metrics and visualization outputs are saved during inference.

Setup

Environment

Python ≥ 3.10

PyTorch ≥ 2.0

CUDA 11.8

Transformers ≥ 4.44

matplotlib, Pillow, tqdm, scikit-learn

Install dependencies:

pip install torch torchvision torchaudio transformers matplotlib pillow tqdm scikit-learn

Training
python train_clipseg_from_txt.py

Testing and Visualization
python test_and_visualize.py


Results (masks and metrics) are saved in:

results/

![Crack segmentation results](results/visual_grid.png)


Notes

This work focuses on prompt generalization and robustness rather than architectural novelty.
Future extensions include multi-defect prompting and comparison with SAM-based segmentation methods.

Repository Structure
.
├── train_clipseg_from_txt.py
├── test_and_visualize.py
├── visual_grid.png
├── results/
└── README.md
