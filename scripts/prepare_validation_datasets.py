#!/usr/bin/env python3
"""prepare_validation_datasets.py

Reorganise raw downloaded COVERAGE and Columbia datasets into the standard
DocForge-Bench layout:

    data/<Dataset>_val/
        images/   ← tampered images
        masks/    ← binary ground-truth masks (same stem as image)

Run after download_validation_datasets.sh:
    cd experiments
    python scripts/prepare_validation_datasets.py

The script is idempotent: re-running it only copies files that are missing.
"""

import shutil
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ─── helpers ─────────────────────────────────────────────────────────────────

def copy_if_missing(src: Path, dst: Path) -> bool:
    """Copy src → dst only if dst does not exist. Returns True if copied."""
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


# ─── COVERAGE ────────────────────────────────────────────────────────────────

def prepare_coverage():
    """
    Source structure (after manual download):
        COVERAGE_raw/COVERAGE/Nt.tif       ← forged images  (e.g. 1t.tif)
        COVERAGE_raw/mask/Nmask.tif        ← GT masks       (e.g. 1mask.tif)

    Target structure:
        COVERAGE_val/images/N.tif
        COVERAGE_val/masks/N.tif
    """
    src_root = DATA_DIR / "COVERAGE_raw"
    img_src = src_root / "COVERAGE"
    mask_src = src_root / "mask"

    if not img_src.is_dir():
        print(f"COVERAGE: source not found ({img_src}) — skipping.")
        print("         Run download_validation_datasets.sh and follow manual instructions.")
        return

    out_root = DATA_DIR / "COVERAGE_val"
    out_img = out_root / "images"
    out_mask = out_root / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    # Forged images: e.g. "1t.tif" → strip trailing 't' → "1.tif"
    images = sorted(img_src.glob("*t.tif"))
    copied_img = copied_mask = 0
    skipped = []

    for img_path in images:
        # stem without trailing 't', e.g. "1t" → "1"
        stem = img_path.stem
        if not stem.endswith("t"):
            # Unexpected naming — skip
            skipped.append(img_path.name)
            continue
        base = stem[:-1]  # strip the 't'
        ext = img_path.suffix

        # Look for corresponding mask: Nmask.tif
        mask_candidates = [
            mask_src / f"{base}mask{ext}",
            mask_src / f"{stem}mask{ext}",   # some versions have "1tmask.tif"
        ]
        mask_path = next((m for m in mask_candidates if m.exists()), None)
        if mask_path is None:
            skipped.append(img_path.name)
            continue

        if copy_if_missing(img_path, out_img / f"{base}{ext}"):
            copied_img += 1
        if copy_if_missing(mask_path, out_mask / f"{base}{ext}"):
            copied_mask += 1

    total_img = len(list(out_img.glob("*")))
    total_mask = len(list(out_mask.glob("*")))
    print(f"COVERAGE: {total_img} images, {total_mask} masks in {out_root}")
    if copied_img:
        print(f"          ({copied_img} images, {copied_mask} masks newly copied)")
    if skipped:
        print(f"          Skipped (no mask found): {', '.join(skipped[:5])}"
              + (f" … +{len(skipped)-5}" if len(skipped) > 5 else ""))


# ─── Columbia ─────────────────────────────────────────────────────────────────

def prepare_columbia():
    """
    Source structure (after download + unzip):
        Columbia_raw/4cam_splc/*.tif              ← forged images
        Columbia_raw/4cam_splc_gt/*_edgemask.tif  ← GT edge masks
        Columbia_raw/4cam_auth/*.tif              ← authentic (optional)

    Target structure:
        Columbia_val/images/X.tif
        Columbia_val/masks/X.tif

    Mask stems: strip "_edgemask" suffix.
    Authentic images are also copied (no corresponding mask → load_dataset
    in run_exp1.py skips them, but they contribute to image-level AUC).
    """
    src_root = DATA_DIR / "Columbia_raw"
    splc_dir = src_root / "4cam_splc"
    gt_dir = src_root / "4cam_splc_gt"
    auth_dir = src_root / "4cam_auth"

    if not splc_dir.is_dir():
        print(f"Columbia: source not found ({splc_dir}) — skipping.")
        print("          Run download_validation_datasets.sh and follow manual instructions.")
        return

    out_root = DATA_DIR / "Columbia_val"
    out_img = out_root / "images"
    out_mask = out_root / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    # ── Spliced images + masks ─────────────────────────────────────────────
    splc_images = sorted(splc_dir.glob("*.tif"))
    copied_img = copied_mask = 0
    skipped = []

    for img_path in splc_images:
        stem = img_path.stem  # e.g. "canong3_canonxt_02"
        ext = img_path.suffix

        # GT mask candidates:
        #   "canong3_canonxt_02_edgemask.tif"  (standard naming)
        #   "canong3_canonxt_02edgemask.tif"   (older version, no underscore)
        mask_candidates = [
            gt_dir / f"{stem}_edgemask{ext}",
            gt_dir / f"{stem}edgemask{ext}",
        ]
        mask_path = next((m for m in mask_candidates if m.exists()), None)
        if mask_path is None:
            skipped.append(img_path.name)
            continue

        if copy_if_missing(img_path, out_img / f"{stem}{ext}"):
            copied_img += 1
        if copy_if_missing(mask_path, out_mask / f"{stem}{ext}"):
            copied_mask += 1

    # ── Authentic images (no mask — skipped by load_dataset) ──────────────
    copied_auth = 0
    if auth_dir.is_dir():
        for img_path in sorted(auth_dir.glob("*.tif")):
            if copy_if_missing(img_path, out_img / img_path.name):
                copied_auth += 1

    total_img = len(list(out_img.glob("*")))
    total_mask = len(list(out_mask.glob("*")))
    print(f"Columbia: {total_img} images ({copied_auth} authentic, "
          f"{total_img - copied_auth} spliced), {total_mask} masks → {out_root}")
    if copied_img:
        print(f"          ({copied_img} spliced, {copied_mask} masks newly copied)")
    if skipped:
        print(f"          Skipped (no mask found): {', '.join(skipped[:5])}"
              + (f" … +{len(skipped)-5}" if len(skipped) > 5 else ""))


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Preparing validation datasets in {DATA_DIR}/")
    print()
    prepare_coverage()
    print()
    prepare_columbia()
    print()

    # Summary
    for name in ("COVERAGE_val", "Columbia_val"):
        d = DATA_DIR / name / "images"
        m = DATA_DIR / name / "masks"
        if d.exists():
            print(f"  ✓ {name}: {len(list(d.glob('*')))} images, "
                  f"{len(list(m.glob('*'))) if m.exists() else 0} masks")
        else:
            print(f"  ✗ {name}: not prepared (manual download required)")

    print()
    print("Next step:")
    print("  python scripts/run_exp1.py --config configs/exp_validation.yaml")


if __name__ == "__main__":
    main()
