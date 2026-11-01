#!/usr/bin/env bash
# download_datasets.sh — Download all benchmark datasets
# Usage: bash download_datasets.sh [--data-dir /path/to/data]
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "============================================"
echo "DocForge-Bench: Dataset Download"
echo "============================================"
echo "Target directory: $DATA_DIR"
echo ""

# ─── Helper ───────────────────────────────────────────────────────────
download_section() {
    local name="$1"
    echo "──────────────────────────────────────────"
    echo "  $name"
    echo "──────────────────────────────────────────"
}

# ═══════════════════════════════════════════════════════════════════════
# A. Document-Specific Tampering
# ═══════════════════════════════════════════════════════════════════════
download_section "A. Document-Specific Tampering"

# --- DocTamper ---
echo "[DocTamper] ~170K document images with pixel-level masks"
echo "  Source: https://github.com/qcf-568/DocTamper"
echo "  Download: Google Drive / Baidu Netdisk links in repo README"
echo "  ACTION REQUIRED: Manual download — visit the GitHub repo for links"
mkdir -p DocTamper
echo "  -> Place files in: $DATA_DIR/DocTamper/"
echo ""

# --- SIDTD ---
echo "[SIDTD] ~25K synthetic identity document images"
echo "  Source: https://tc11.cvc.uab.es/datasets/SIDTD_1"
echo "  ACTION REQUIRED: Register and download from the TC11 website"
mkdir -p SIDTD
echo "  -> Place files in: $DATA_DIR/SIDTD/"
echo ""

# --- MIDV-LAIT ---
echo "[MIDV-LAIT] ~1K identity document tampering images"
echo "  Source: https://github.com/SmartEngines/midv-lait"
mkdir -p MIDV-LAIT
if [ ! -f "MIDV-LAIT/.downloaded" ]; then
    git clone --depth 1 https://github.com/SmartEngines/midv-lait.git MIDV-LAIT/repo 2>/dev/null \
        && touch MIDV-LAIT/.downloaded \
        || echo "  [WARN] Failed to clone MIDV-LAIT. Download manually."
else
    echo "  [SKIP] Already downloaded"
fi
echo ""

# --- MIDV-2020 ---
echo "[MIDV-2020] 1K video clips of identity documents"
echo "  Source: ftp://ftp.smartengines.com/midv-2020/"
echo "  ACTION REQUIRED: Download via FTP client"
mkdir -p MIDV-2020
echo "  -> Place files in: $DATA_DIR/MIDV-2020/"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# B. Signature Verification
# ═══════════════════════════════════════════════════════════════════════
download_section "B. Signature Verification"

# --- CEDAR ---
echo "[CEDAR] 2,640 signatures (55 writers)"
echo "  Source: http://www.cedar.buffalo.edu/NIJ/Publications/"
echo "  ACTION REQUIRED: Request access from CEDAR lab"
mkdir -p CEDAR
echo "  -> Place files in: $DATA_DIR/CEDAR/"
echo ""

# --- BHSig260 ---
echo "[BHSig260] ~15K signatures (Bengali + Hindi)"
echo "  Source: Available on Kaggle (search 'BHSig260')"
echo "  ACTION REQUIRED: Download from Kaggle"
mkdir -p BHSig260
echo "  -> Place files in: $DATA_DIR/BHSig260/"
echo ""

# --- GPDS-960 ---
echo "[GPDS-960] ~47K signatures"
echo "  Source: Formal request required from ULPGC"
echo "  ACTION REQUIRED: Request from researchers"
mkdir -p GPDS-960
echo "  -> Place files in: $DATA_DIR/GPDS-960/"
echo ""

# --- UTSig ---
echo "[UTSig] ~5K Persian signatures"
echo "  Source: University of Tehran"
echo "  ACTION REQUIRED: Request access"
mkdir -p UTSig
echo "  -> Place files in: $DATA_DIR/UTSig/"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# C. General Image Forgery
# ═══════════════════════════════════════════════════════════════════════
download_section "C. General Image Forgery"

# --- CASIA v2 ---
echo "[CASIA v2] ~12K images with masks"
echo "  Source: Available on Kaggle (search 'CASIA v2')"
echo "  ACTION REQUIRED: Download from Kaggle"
mkdir -p CASIAv2
echo "  -> Place files in: $DATA_DIR/CASIAv2/"
echo ""

# --- Columbia ---
echo "[Columbia] ~1.8K uncompressed splicing images"
echo "  Source: https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/"
echo "  ACTION REQUIRED: Download from Columbia website"
mkdir -p Columbia
echo "  -> Place files in: $DATA_DIR/Columbia/"
echo ""

# --- COVERAGE ---
echo "[COVERAGE] 200 copy-move image pairs"
echo "  Source: https://github.com/wenbihan/coverage"
mkdir -p COVERAGE
if [ ! -f "COVERAGE/.downloaded" ]; then
    git clone --depth 1 https://github.com/wenbihan/coverage.git COVERAGE/repo 2>/dev/null \
        && touch COVERAGE/.downloaded \
        || echo "  [WARN] Failed to clone COVERAGE. Download manually."
else
    echo "  [SKIP] Already downloaded"
fi
echo ""

# --- CoMoFoD ---
echo "[CoMoFoD] 260 image sets"
echo "  Source: https://www.vcl.fer.hr/comofod/"
echo "  ACTION REQUIRED: Download from VCL website"
mkdir -p CoMoFoD
echo "  -> Place files in: $DATA_DIR/CoMoFoD/"
echo ""

# --- IMD2020 ---
echo "[IMD2020] 2K manipulated images"
echo "  Source: https://staff.utia.cas.cz/novozada/db/"
echo "  ACTION REQUIRED: Download from UTIA website"
mkdir -p IMD2020
echo "  -> Place files in: $DATA_DIR/IMD2020/"
echo ""

# --- NIST NC16 ---
echo "[NIST NC16] ~564 images"
echo "  Source: NIST (data agreement required)"
echo "  ACTION REQUIRED: Apply for access via NIST"
mkdir -p NIST_NC16
echo "  -> Place files in: $DATA_DIR/NIST_NC16/"
echo ""

# --- DEFACTO ---
echo "[DEFACTO] ~229K images with masks"
echo "  Source: https://defactodataset.github.io/"
echo "  ACTION REQUIRED: Download from DEFACTO website"
mkdir -p DEFACTO
echo "  -> Place files in: $DATA_DIR/DEFACTO/"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# D. AI-Generated Content
# ═══════════════════════════════════════════════════════════════════════
download_section "D. AI-Generated Content"

# --- GenImage ---
echo "[GenImage] 1.3M+ AI-generated images"
echo "  Source: https://github.com/GenImage-Dataset/GenImage"
echo "  ACTION REQUIRED: Follow download instructions in repo README"
mkdir -p GenImage
echo "  -> Place files in: $DATA_DIR/GenImage/"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
echo "============================================"
echo "Dataset directories created in: $DATA_DIR"
echo ""
echo "Most datasets require manual download due to"
echo "licensing, registration, or access agreements."
echo ""
echo "After downloading, verify structure with:"
echo "  ls -la $DATA_DIR/*/"
echo "============================================"
