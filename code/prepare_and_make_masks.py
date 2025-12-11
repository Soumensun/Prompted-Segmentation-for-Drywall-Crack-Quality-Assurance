# prepare_and_make_masks.py
import os, sys, shutil, re
from pathlib import Path
from PIL import Image, ImageDraw

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def find_files(root):
    imgs, txts = [], []
    for p in Path(root).rglob("*"):
        if p.is_file():
            if p.suffix.lower() in IMAGE_EXTS:
                imgs.append(str(p))
            elif p.suffix.lower() == ".txt":
                txts.append(str(p))
    return imgs, txts

def normalize_base(name):
    # remove Roboflow suffixes like ".rf.<hash>" and keep part before extension
    # Examples:
    # "1_0005_2-Vertical-cracks_png_jpg.rf.0cf48..." -> "1_0005_2-Vertical-cracks_png_jpg"
    # "00002.jpg" -> "00002"
    base = Path(name).stem
    # Remove ".rf.<hash>" or ".rf<hash>" patterns if present
    base = re.sub(r"\.rf\.[0-9a-fA-F]+$|\.rf[0-9a-fA-F]+$", "", base)
    # Remove common double-suffix like "_png_jpg" -> keep as-is (we'll match images/texts by base)
    base = re.sub(r"_png$|_jpg$|_jpeg$", "", base)
    return base

def build_lookup(paths):
    lookup = {}
    for p in paths:
        base = normalize_base(os.path.basename(p))
        lookup.setdefault(base, []).append(p)
    return lookup

def pair_images_labels(img_lookup, txt_lookup):
    pairs = []
    for base, img_paths in img_lookup.items():
        # pick first image path for this base (if multiple)
        img_path = sorted(img_paths)[0]
        # find matching txts: exact base match or any txts whose normalize_base equals base
        txt_candidates = txt_lookup.get(base, [])
        if not txt_candidates:
            # try fuzzy matching by checking txts whose normalized base contains our base or vice versa
            for tb, paths in txt_lookup.items():
                if base in tb or tb in base:
                    txt_candidates.extend(paths)
        txt_path = sorted(txt_candidates)[0] if txt_candidates else None
        pairs.append((img_path, txt_path))
    return pairs

def ensure_dirs(root_out):
    images_out = os.path.join(root_out, "images")
    labels_out = os.path.join(root_out, "labels")
    masks_out = os.path.join(root_out, "masks")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)
    return images_out, labels_out, masks_out

def copy_and_rename(pairs, images_out, labels_out):
    mapping_lines = []
    for i, (img, txt) in enumerate(sorted(pairs), start=1):
        id_str = f"{i:05d}"
        ext = Path(img).suffix.lower()
        new_img = os.path.join(images_out, id_str + ext)
        shutil.copy2(img, new_img)
        new_txt = None
        if txt:
            new_txt = os.path.join(labels_out, id_str + ".txt")
            shutil.copy2(txt, new_txt)
        mapping_lines.append((new_img, new_txt))
    return mapping_lines

def yolo_txt_to_mask(txt_path, img_w, img_h):
    """
    Parse a YOLO label file. Accepts:
    - 5 values per line: class xc yc w h (normalized) -> rectangle
    - >5 even values: polygon coords normalized (skips first class token)
    Returns a PIL L-mode mask.
    """
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # skip class
            vals = list(map(float, parts[1:])) if len(parts) > 5 else list(map(float, parts[1:5]))
            if len(vals) == 4:
                xc, yc, bw, bh = vals
                x1 = (xc - bw/2) * img_w
                y1 = (yc - bh/2) * img_h
                x2 = (xc + bw/2) * img_w
                y2 = (yc + bh/2) * img_h
                draw.rectangle([x1, y1, x2, y2], fill=255)
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                coords = []
                for j in range(0, len(vals), 2):
                    x = vals[j] * img_w
                    y = vals[j+1] * img_h
                    coords.append((x,y))
                if len(coords) >= 3:
                    draw.polygon(coords, fill=255)
    return mask

def generate_masks(mapping_lines, masks_out):
    print("Generating masks...")
    for img_path, txt_path in mapping_lines:
        img = Image.open(img_path)
        w,h = img.size
        base = Path(img_path).stem
        out_mask_p = os.path.join(masks_out, base + ".png")
        if txt_path and os.path.exists(txt_path):
            mask = yolo_txt_to_mask(txt_path, w, h)
            mask.save(out_mask_p)
        else:
            # Save empty mask (all zeros) if no label file
            Image.new("L", (w,h), 0).save(out_mask_p)
        print("Saved mask", out_mask_p)

def main():
    if len(sys.argv) < 3:
        print("Usage: python prepare_and_make_masks.py <input_root_folder> <output_data_root>")
        print("Example: python prepare_and_make_masks.py ./cracks-3ii36-1/train ./data/cracks")
        sys.exit(1)
    src_root = sys.argv[1]
    out_root = sys.argv[2]

    print("Scanning for images/labels under:", src_root)
    imgs, txts = find_files(src_root)
    print(f"Found {len(imgs)} image files and {len(txts)} txt label files.")
    if len(imgs) == 0:
        print("No images found. Check the input folder path.")
        sys.exit(1)

    img_lookup = build_lookup(imgs)
    txt_lookup = build_lookup(txts)
    pairs = pair_images_labels(img_lookup, txt_lookup)
    images_out, labels_out, masks_out = ensure_dirs(out_root)
    mapping = copy_and_rename(pairs, images_out, labels_out)

    # write mapping file
    list_path = os.path.join(out_root, "file_list.txt")
    with open(list_path, "w") as f:
        for img_p, txt_p in mapping:
            rel_img = os.path.relpath(img_p, out_root)
            rel_txt = os.path.relpath(txt_p, out_root) if txt_p else "NONE"
            f.write(f"{rel_img} {rel_txt}\n")
    print("Wrote mapping to", list_path)

    # generate masks automatically
    generate_masks(mapping, masks_out)
    print("Done. Images ->", images_out)
    print("Labels ->", labels_out)
    print("Masks ->", masks_out)
    print("If you want different naming, edit the script. Now you can run training using data in:", out_root)

if __name__ == "__main__":
    main()
