#!/bin/bash
# Assembles a 60-way (identity x emotion) dataset via symlinks, with an equal
# number of images per class, from existing render batches on the locker:
#   - 56way_IDEM_colorbg: 8 identities (Ani, Dan, Kedar, Seojin, Sophie, Sreyas,
#     Tony, Younah) x 7 emotions = 56 classes, already train/val split
#   - texture_colorbg_2way_elias_neptune: elias_neutral, neptune_neutral,
#     already train/val split
#   - vbsl50k_ashley_neutral, vbsl50k_josh_neutral: ashley_neutral,
#     josh_neutral, flat pools of images (no train/val split -- we split them
#     here)
#
# Target count is capped by the smallest available class (56way's ~1430
# train / 357 val per class); N_TRAIN/N_VAL below are set safely under that.

set -eo pipefail

SRC_BASE="/mnt/smb/locker/issa-locker/users/Seojin/data/face_data/vbsle_50k_texture_colorbg"
OUT_BASE="$SRC_BASE/60way_IDEM_colorbg"
N_TRAIN=1400
N_VAL=350

mkdir -p "$OUT_BASE/train" "$OUT_BASE/val"

echo "--- 56 existing identity x emotion classes ---"
for split in train val; do
  n=$N_TRAIN
  [ "$split" = "val" ] && n=$N_VAL
  for classdir in "$SRC_BASE/56way_IDEM_colorbg/$split"/*/; do
    cls=$(basename "$classdir")
    mkdir -p "$OUT_BASE/$split/$cls"
    find "$classdir" -maxdepth 1 -type f | sort | head -n "$n" | while read -r f; do
      ln -sf "$f" "$OUT_BASE/$split/$cls/$(basename "$f")"
    done
    echo "  $split/$cls done"
  done
done

echo "--- elias_neutral, neptune_neutral (already train/val split) ---"
for pair in "elias_neutral:elias" "neptune_neutral:neptune"; do
  cls="${pair%%:*}"
  src_name="${pair##*:}"
  mkdir -p "$OUT_BASE/train/$cls" "$OUT_BASE/val/$cls"
  find "$SRC_BASE/texture_colorbg_2way_elias_neptune/train/$src_name" -maxdepth 1 -type f | sort | head -n "$N_TRAIN" | while read -r f; do
    ln -sf "$f" "$OUT_BASE/train/$cls/$(basename "$f")"
  done
  find "$SRC_BASE/texture_colorbg_2way_elias_neptune/val/$src_name" -maxdepth 1 -type f | sort | head -n "$N_VAL" | while read -r f; do
    ln -sf "$f" "$OUT_BASE/val/$cls/$(basename "$f")"
  done
  echo "  $cls done"
done

echo "--- ashley_neutral, josh_neutral (flat pools, splitting train/val ourselves) ---"
for pair in "ashley_neutral:vbsl50k_ashley_neutral" "josh_neutral:vbsl50k_josh_neutral"; do
  cls="${pair%%:*}"
  src_dir="$SRC_BASE/${pair##*:}"
  mkdir -p "$OUT_BASE/train/$cls" "$OUT_BASE/val/$cls"
  filelist="/tmp/${cls}_files.txt"
  find "$src_dir" -maxdepth 1 -type f | sort > "$filelist"
  head -n "$N_TRAIN" "$filelist" | while read -r f; do
    ln -sf "$f" "$OUT_BASE/train/$cls/$(basename "$f")"
  done
  tail -n +$((N_TRAIN + 1)) "$filelist" | head -n "$N_VAL" | while read -r f; do
    ln -sf "$f" "$OUT_BASE/val/$cls/$(basename "$f")"
  done
  rm -f "$filelist"
  echo "  $cls done"
done

echo "--- verifying per-class counts ---"
for split in train val; do
  for d in "$OUT_BASE/$split"/*/; do
    cnt=$(find "$d" -maxdepth 1 -type l | wc -l)
    echo "$split/$(basename "$d"): $cnt"
  done
done
