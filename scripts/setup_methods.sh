#!/usr/bin/env bash
# setup_methods.sh — Recreate methods/ exactly as used for the paper.
#
# For each of the 11 upstream repositories (14 methods; ForensicHub provides DTD, FFDN,
# CAFTB-Net and TIFDM) this script:
#   1. clones the repository and checks out the commit we evaluated (methods/pins.txt),
#   2. applies our small compatibility patch, if any (methods/patches/<repo>.patch),
#   3. copies our inference wrapper(s) into the clone (methods/wrappers/<repo>/).
#
# It does NOT download pretrained weights: each method's weights come from its authors
# under their own terms. methods/README.md lists every file and where it must be placed.
#
# Usage (from the repository root):
#   bash scripts/setup_methods.sh            # all methods
#   bash scripts/setup_methods.sh TruFor RTM # selected repositories
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHODS="$ROOT/methods"
PINS="$METHODS/pins.txt"

want() {
    [ $# -eq 0 ] && return 0
    local n
    for n in "${SELECTED[@]}"; do [ "$n" = "$1" ] && return 0; done
    return 1
}
SELECTED=("$@")

while IFS='|' read -r name url commit; do
    [ -z "$name" ] && continue
    if [ ${#SELECTED[@]} -gt 0 ] && ! want "$name"; then continue; fi
    dest="$METHODS/$name"
    echo "==> $name @ ${commit:0:7}"
    if [ ! -d "$dest/.git" ]; then
        git clone --quiet "$url" "$dest"
    fi
    git -C "$dest" fetch --quiet origin "$commit" 2>/dev/null || true
    git -C "$dest" checkout --quiet "$commit"

    patch="$METHODS/patches/$name.patch"
    if [ -f "$patch" ]; then
        if git -C "$dest" apply --check "$patch" 2>/dev/null; then
            git -C "$dest" apply "$patch"
            echo "    applied patches/$name.patch"
        else
            echo "    patches/$name.patch already applied or does not apply cleanly; skipped"
        fi
    fi

    if [ -d "$METHODS/wrappers/$name" ]; then
        cp -r "$METHODS/wrappers/$name/." "$dest/"
        echo "    installed wrappers/$name"
    fi
done < "$PINS"

echo
echo "Done. Next: place each method's pretrained weights as listed in methods/README.md,"
echo "and create the per-method conda environments described there."
