# train_clipseg_from_txt.py
import os, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from sklearn.metrics import f1_score, jaccard_score
from tqdm import tqdm

#  Dataset 
class ListDataset(Dataset):
    def __init__(self, txt_file, root, prompt):
        self.samples = []
        with open(txt_file) as f:
            for line in f:
                img_rel, mask_rel = line.strip().split()
                self.samples.append((root/img_rel, root/mask_rel))
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.prompt = prompt
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        proc = self.processor(text=self.prompt, images=image, return_tensors="pt")
        _, _, H, W = proc["pixel_values"].shape
        mask = Image.open(mask_path).convert("L").resize((W,H), Image.NEAREST)
        mask = (np.array(mask)>127).astype("float32")
        proc["labels"] = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return proc

def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.cat([b[k] for b in batch], dim=0)
    return out


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
root = torch.tensor  # dummy to satisfy syntax highlighter
from pathlib import Path
ROOT = Path("data/cracks")
PROMPT = "segment crack"
BATCH = 2
EPOCHS = 3
LR = 5e-5

trainset = ListDataset(ROOT/"train.txt", ROOT, PROMPT)
valset = ListDataset(ROOT/"val.txt", ROOT, PROMPT)
testset = ListDataset(ROOT/"test.txt", ROOT, PROMPT)


trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
valloader   = DataLoader(valset, batch_size=1, shuffle=False, collate_fn=collate_fn)
testloader  = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

# -------- Training --------
for epoch in range(EPOCHS):
    model.train(); running = 0.0
    for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch = {k:v.to(device) for k,v in batch.items()}
        loss = model(**batch).loss
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()
    print(f"Train loss: {running/len(trainloader):.4f}")

    # quick val dice
    model.eval(); dice, iou = [], []
    with torch.no_grad():
        for batch in valloader:
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch).logits
            pred = (torch.sigmoid(out)>0.5).float().cpu().numpy()
            gt = batch["labels"].cpu().numpy()
            for p,g in zip(pred,gt):
                dice.append(f1_score(g.flatten(), p.flatten()))
                iou.append(jaccard_score(g.flatten(), p.flatten()))
    print(f"Val Dice {np.mean(dice):.3f} | mIoU {np.mean(iou):.3f}")

#  Test evaluation 
model.eval(); dice, iou = [], []
with torch.no_grad():
    for batch in testloader:
        batch = {k:v.to(device) for k,v in batch.items()}
        out = model(**batch).logits
        pred = (torch.sigmoid(out)>0.5).float().cpu().numpy()
        gt = batch["labels"].cpu().numpy()
        for p,g in zip(pred,gt):
            dice.append(f1_score(g.flatten(), p.flatten()))
            iou.append(jaccard_score(g.flatten(), p.flatten()))
print(f" Test Dice {np.mean(dice):.3f} | Test mIoU {np.mean(iou):.3f}")