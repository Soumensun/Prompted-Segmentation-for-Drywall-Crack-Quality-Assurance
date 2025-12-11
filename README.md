#  Prompted Segmentation for Drywall & Crack QA
**Author:** Soumen  
**Institute:** IIT Bhubaneswar  
**Date:** November 2025  

---

##  Project Overview
This project fine-tunes a **text-conditioned segmentation model (CLIPSeg)** to detect wall cracks and drywall taping areas.  
The goal is to evaluate how well a *prompt-based* model can generalize across different construction scenes when given natural-language commands like:

- “segment crack”  
- “segment taping area”

The model learns to highlight relevant regions in each image — useful for autonomous inspection robots on job sites.

---

##  Setup Instructions
### 1. Environment
Tested on:
- Python 3.10+
- PyTorch 2.x
- CUDA 11.8
- Transformers (Hugging Face) 4.44+
- scikit-learn, matplotlib, Pillow, tqdm  

Install everything quickly:
```bash
pip install torch torchvision torchaudio transformers scikit-learn matplotlib pillow tqdm

#Train the model:
-python train_clipseg_from_txt.py

#Test & visualize:
-python test_and_visualize.py

#Results (Dice, mIoU, and visuals) are saved in results/.
