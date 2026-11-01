#!/usr/bin/env bash
# download_validation_datasets.sh
# Download COVERAGE and Columbia datasets for paper-vs-benchmark validation.
#
# Columbia:  Direct academic download (blocked by Cloudflare bot protection as of 2026-02).
#            Falls back to manual instructions.
# COVERAGE:  Only available via OneDrive / BaiduYun — manual download required.
#
# Usage:
#   cd experiments
#   bash scripts/download_validation_datasets.sh
#
# Output directories:
#   data/Columbia_raw/   — raw zip files + extracted 4cam_* subdirs
#   data/COVERAGE_raw/   — extracted COVERAGE/ and mask/ subdirs

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/../data" && pwd)"

COLUMBIA_DIR="$DATA_DIR/Columbia_raw"
COVERAGE_DIR="$DATA_DIR/COVERAGE_raw"

# ─── Columbia ────────────────────────────────────────────────────────────────
echo ""
echo "=== Columbia Uncompressed Image Splicing Detection Dataset ==="

COLUMBIA_OK=false
mkdir -p "$COLUMBIA_DIR"

if [[ -d "$COLUMBIA_DIR/4cam_splc" && -d "$COLUMBIA_DIR/4cam_splc_gt" ]]; then
    SPLC_COUNT=$(find "$COLUMBIA_DIR/4cam_splc" -name "*.tif" | wc -l)
    GT_COUNT=$(find "$COLUMBIA_DIR/4cam_splc_gt" -name "*edgemask*" | wc -l)
    if [[ "$SPLC_COUNT" -gt 100 && "$GT_COUNT" -gt 100 ]]; then
        echo "  Already present: $SPLC_COUNT spliced images, $GT_COUNT GT masks"
        COLUMBIA_OK=true
    fi
fi

if [[ "$COLUMBIA_OK" == "false" ]]; then
    BASE="https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp"
    echo "  Attempting download from Columbia DVMM lab..."
    cd "$COLUMBIA_DIR"

    DOWNLOAD_OK=true
    for ZIP in 4cam_auth.zip 4cam_splc.zip 4cam_splc_gt.zip; do
        if [[ ! -f "$ZIP" ]]; then
            echo "  Downloading $ZIP ..."
            if ! wget -q --timeout=30 --tries=2 "$BASE/$ZIP" 2>/dev/null; then
                echo "  WARNING: Failed to download $ZIP"
                DOWNLOAD_OK=false
                break
            fi
        fi
    done

    if [[ "$DOWNLOAD_OK" == "true" ]]; then
        echo "  Extracting..."
        unzip -q -o 4cam_auth.zip
        unzip -q -o 4cam_splc.zip
        unzip -q -o 4cam_splc_gt.zip
        echo "  Columbia downloaded and extracted OK."
        COLUMBIA_OK=true
    fi
    cd "$SCRIPT_DIR/.."
fi

if [[ "$COLUMBIA_OK" == "false" ]]; then
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────────┐"
    echo "  │  ACTION REQUIRED: Manual Columbia download                      │"
    echo "  │                                                                 │"
    echo "  │  The Columbia DVMM server uses Cloudflare bot protection that   │"
    echo "  │  blocks automated wget/curl downloads.                          │"
    echo "  │                                                                 │"
    echo "  │  Option A — Direct download (browser):                          │"
    echo "  │    1. Visit: https://www.ee.columbia.edu/ln/dvmm/downloads/    │"
    echo "  │              authsplcuncmp/                                     │"
    echo "  │    2. Download 4cam_splc.zip and 4cam_splc_gt.zip              │"
    echo "  │    3. Extract into: data/Columbia_raw/             │"
    echo "  │                                                                 │"
    echo "  │  Option B — Kaggle mirror (search 'columbia splicing forgery'): │"
    echo "  │    kaggle datasets download -d <user>/columbia-splicing-dataset │"
    echo "  │    mv columbia-splicing-dataset data/Columbia_raw/  │"
    echo "  │                                                                 │"
    echo "  │  Expected structure after extraction:                           │"
    echo "  │    data/Columbia_raw/4cam_splc/*.tif        (180 images)       │"
    echo "  │    data/Columbia_raw/4cam_splc_gt/*_edgemask.tif  (180 masks) │"
    echo "  └─────────────────────────────────────────────────────────────────┘"
fi

# ─── COVERAGE ────────────────────────────────────────────────────────────────
echo ""
echo "=== COVERAGE Copy-Move Forgery Dataset ==="

COVERAGE_OK=false
mkdir -p "$COVERAGE_DIR"

if [[ -d "$COVERAGE_DIR/COVERAGE" && -d "$COVERAGE_DIR/mask" ]]; then
    IMG_COUNT=$(find "$COVERAGE_DIR/COVERAGE" -name "*t.tif" | wc -l)
    MASK_COUNT=$(find "$COVERAGE_DIR/mask" -name "*mask.tif" | wc -l)
    if [[ "$IMG_COUNT" -gt 50 && "$MASK_COUNT" -gt 50 ]]; then
        echo "  Already present: $IMG_COUNT tampered images, $MASK_COUNT masks"
        COVERAGE_OK=true
    fi
fi

if [[ "$COVERAGE_OK" == "false" ]]; then
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────────┐"
    echo "  │  ACTION REQUIRED: Manual COVERAGE download                      │"
    echo "  │                                                                 │"
    echo "  │  COVERAGE is only available via OneDrive or BaiduYun:           │"
    echo "  │                                                                 │"
    echo "  │  Option A — OneDrive:                                           │"
    echo "  │    https://1drv.ms/f/s!AggVhXcCj1FLhUUyUrqSpV_yI_GH            │"
    echo "  │                                                                 │"
    echo "  │  Option B — BaiduYun (China users):                             │"
    echo "  │    https://pan.baidu.com/s/11i_swrFveLc9uZr1eR006Q             │"
    echo "  │    Password: <request from dataset authors>                                             │"
    echo "  │                                                                 │"
    echo "  │  After downloading, extract so the structure is:                │"
    echo "  │    data/COVERAGE_raw/COVERAGE/*t.tif   (100 forged images)     │"
    echo "  │    data/COVERAGE_raw/mask/*mask.tif    (100 GT masks)          │"
    echo "  │    data/COVERAGE_raw/Origin/*o.tif     (100 original images)   │"
    echo "  └─────────────────────────────────────────────────────────────────┘"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Download Summary ==="
echo "  Columbia: $([ $COLUMBIA_OK == true ] && echo '✓ READY' || echo '✗ MANUAL DOWNLOAD NEEDED')"
echo "  COVERAGE: $([ $COVERAGE_OK == true ] && echo '✓ READY' || echo '✗ MANUAL DOWNLOAD NEEDED')"
echo ""
echo "  After manual downloads, run:"
echo "    python scripts/prepare_validation_datasets.py"
echo "    python scripts/run_exp1.py --config configs/exp_validation.yaml"
