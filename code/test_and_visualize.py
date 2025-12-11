# test_and_visualize.py
# Run evaluation on the test split and save visual plots + CSV.
import os, csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from sklearn.metrics import f1_score, jaccard_score
import matplotlib.pyplot as plt

# ----------------- USER CONFIG -----------------
ROOT = Path("data/cracks")                     # root where train.txt/val.txt/test.txt live
TEST_LIST = ROOT / "test.txt"
WEIGHTS_PATH = Path("weights/clipseg_crack_split.pth")  # change if different
PROMPT = "segment crack"
BATCH = 1
THRESH = 0.5
NUM_VIS_EXAMPLES = 12   # how many image rows to show in visual grid
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True, parents=True)
# ------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---------- dataset that reads test.txt ----------
class ListTestDataset(Dataset):
    def __init__(self, list_path, root, prompt):
        self.samples = []
        with open(list_path, "r") as f:
            for line in f:
                img_rel, mask_rel = line.strip().split()
                self.samples.append((root / img_rel, root / mask_rel))
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.prompt = prompt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        proc = self.processor(text=self.prompt, images=image, return_tensors="pt")
        _, _, H, W = proc["pixel_values"].shape
        mask = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
        mask_arr = (np.array(mask) > 127).astype("uint8")
       
        proc["labels"] = torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(0)  # (1,H,W) float32

        # attach paths for bookkeeping
        proc["img_path"] = str(img_path)
        proc["mask_path"] = str(mask_path)
        return proc

def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        # skip non-tensor keys (img_path, mask_path) when concatenating
        if k in ("img_path","mask_path"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.cat([b[k] for b in batch], dim=0)
    return out

# ---------- load dataset ----------
testset = ListTestDataset(TEST_LIST, ROOT, PROMPT)
testloader = DataLoader(testset, batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

# ---------- load model ----------
print("Loading model...")
try:
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    if WEIGHTS_PATH.exists():
        print(f"Loading fine-tuned weights from {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model = model.to(device)
except Exception as e:
    raise RuntimeError("Model load failed: " + str(e))

model.eval()

# ---------- evaluation loop ----------
rows = []
dice_list = []
iou_list = []
vis_examples = []  # list of tuples (PIL_img, GT_mask, Pred_mask, img_path)

with torch.no_grad():
    for batch_idx, batch in enumerate(testloader):
        # move tensors
        inputs = {}
        for k,v in batch.items():
            if k in ("img_path","mask_path"): continue
            inputs[k] = v.to(device)
        logits = model(**inputs).logits  # logits shape (B, 1, H, W) or (B, H, W)
        # normalize logits shape to (B, H, W)
        if logits.dim() == 4 and logits.shape[1] == 1:
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        else:
            probs = torch.sigmoid(logits).cpu().numpy()
        gts = batch["labels"].cpu().numpy()  # (B,1,H,W) or (B,H,W)
        if gts.ndim == 4 and gts.shape[1] == 1:
            gts = gts.squeeze(1)
        # iterate items in batch
        img_paths = batch["img_path"]
        mask_paths = batch["mask_path"]
        for i in range(probs.shape[0]):
            prob = probs[i]
            pred_bin = (prob > THRESH).astype("uint8")
            gt = gts[i].astype("uint8")
            # flatten for metrics
            p_flat = pred_bin.flatten()
            g_flat = gt.flatten()
            # handle degenerate case: if gt all zeros (no defect), f1/jaccard could be ill-defined.
            try:
                dice = f1_score(g_flat, p_flat, zero_division=1)
                iou = jaccard_score(g_flat, p_flat, zero_division=1)
            except Exception:
                # fallback: compute manually
                inter = (p_flat & g_flat).sum()
                union = (p_flat | g_flat).sum()
                dice = (2*inter) / (p_flat.sum() + g_flat.sum() + 1e-8)
                iou = inter / (union + 1e-8)
            dice_list.append(dice)
            iou_list.append(iou)
            rows.append({
                "img": img_paths[i],
                "mask": mask_paths[i],
                "dice": float(dice),
                "iou": float(iou)
            })
            # store first NUM_VIS_EXAMPLES for visual grid
            if len(vis_examples) < NUM_VIS_EXAMPLES:
                # load original image at original resolution for visualization
                orig_img = Image.open(img_paths[i]).convert("RGB")
                # load GT mask at original size and also pred mask resized back to original size
                gt_mask_full = Image.open(mask_paths[i]).convert("L")
                pred_mask_img = Image.fromarray((pred_bin * 255).astype("uint8"))
                # resize pred mask to original image size
                pred_mask_full = pred_mask_img.resize(orig_img.size, Image.NEAREST)
                vis_examples.append((orig_img, gt_mask_full, pred_mask_full, img_paths[i]))

# ---------- save CSV ----------
csv_path = OUT_DIR / "results.csv"
with open(csv_path, "w", newline="") as csvfile:
    fieldnames = ["img", "mask", "dice", "iou"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
print(f"Saved metrics CSV to: {csv_path}")

# ---------- summary ----------
dice_arr = np.array(dice_list)
iou_arr = np.array(iou_list)
print(f"Test samples: {len(dice_arr)}")
print(f"Dice (mean ± std): {dice_arr.mean():.4f} ± {dice_arr.std():.4f}")
print(f"mIoU (mean ± std): {iou_arr.mean():.4f} ± {iou_arr.std():.4f}")

# ---------- visual grid ----------
def make_visual_grid(vis_list, ncols=3, h_per_row=256):
    n = len(vis_list)
    if n == 0:
        return None
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    # each row contains (orig | GT | Pred) vertically stacked
    # we'll create a grid of nrows x ncols columns where each cell is a triple stacked vertically
    # compute width/height from first image
    w, h = vis_list[0][0].size
    cell_w = w
    cell_h = h
    # we'll stack images horizontally per example: orig, gt, pred in a single cell width*3
    cell_total_w = cell_w * 3
    grid_w = ncols * cell_total_w
    grid_h = nrows * cell_h
    grid = Image.new("RGB", (grid_w, grid_h), color=(255,255,255))
    for idx, (orig, gt, pred, path) in enumerate(vis_list):
        row = idx // ncols
        col = idx % ncols
        x0 = col * cell_total_w
        y0 = row * cell_h
        # prepare GT and pred as RGB for display
        gt_rgb = Image.merge("RGB", (gt, gt, gt)).convert("RGB")
        pred_rgb = Image.merge("RGB", (pred, pred, pred)).convert("RGB")
        # paste orig | gt | pred side by side
        grid.paste(orig, (x0 + 0*cell_w, y0))
        grid.paste(gt_rgb, (x0 + 1*cell_w, y0))
        grid.paste(pred_rgb, (x0 + 2*cell_w, y0))
    return grid

grid = make_visual_grid(vis_examples, ncols=3)
if grid:
    grid_path = OUT_DIR / "visual_grid.png"
    grid.save(grid_path)
    print(f"Saved visual grid to: {grid_path}")
else:
    print("No visual examples to save.")

# ---------- dice histogram ----------
plt.figure(figsize=(6,4))
plt.hist(dice_arr, bins=20, edgecolor="k", alpha=0.7)
plt.title("Dice score distribution (test set)")
plt.xlabel("Dice")
plt.ylabel("Count")
plt.grid(alpha=0.3)
hist_path = OUT_DIR / "dice_hist.png"
plt.tight_layout()
plt.savefig(hist_path, dpi=150)
plt.close()
print(f"Saved histogram to: {hist_path}")

# ---------- dice vs iou scatter ----------
plt.figure(figsize=(5,5))
plt.scatter(dice_arr, iou_arr, alpha=0.6)
plt.xlabel("Dice")
plt.ylabel("mIoU")
plt.title("Dice vs mIoU (test set)")
plt.grid(alpha=0.3)
scatter_path = OUT_DIR / "dice_scatter.png"
plt.tight_layout()
plt.savefig(scatter_path, dpi=150)
plt.close()
print(f"Saved scatter to: {scatter_path}")

print("All done. Results and plots are in the 'results' folder.")
