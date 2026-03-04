#!/usr/bin/env bash
# migrate_to_k.sh — Continue moving data to /Volumes/k and clean up
# Resumable: rsync only copies files that don't exist on K yet
set -e

SRC="/Users/ahmed/Desktop/thesis/My-Master-thesis/data/real_world_10k"
DST="/Volumes/k/thesis_data/real_world_10k"

echo "=== Resumable migration to external disk K ==="
echo "  Source: $SRC"
echo "  Dest:   $DST"
echo ""

# 1. rsync remaining files (skips already-copied ones)
echo "[1/4] Syncing remaining files to K (this may take a while)..."
rsync -a --progress "$SRC/" "$DST/"
echo "  Sync complete."

# 2. Verify file counts match
SRC_COUNT=$(find "$SRC/raw" -type f | wc -l | tr -d ' ')
DST_COUNT=$(find "$DST/raw" -type f | wc -l | tr -d ' ')
echo ""
echo "[2/4] Verifying file counts..."
echo "  Internal: $SRC_COUNT files"
echo "  External: $DST_COUNT files"

if [[ "$SRC_COUNT" != "$DST_COUNT" ]]; then
    echo "  ERROR: File counts don't match! Aborting delete."
    exit 1
fi
echo "  Counts match."

# 3. Delete internal copy
echo ""
echo "[3/4] Removing internal copy to free ~10 GB..."
rm -rf "$SRC"
echo "  Deleted $SRC"

# 4. Create symlink so all scripts still work with same path
echo ""
echo "[4/4] Creating symlink..."
ln -s "$DST" "$SRC"
echo "  $SRC -> $DST"

echo ""
echo "=== DONE ==="
echo "  Data lives on: $DST"
echo "  Symlink:       $SRC -> $DST"
echo "  All scripts work unchanged."
df -h /Volumes/k | tail -1
df -h /System/Volumes/Data | tail -1
 