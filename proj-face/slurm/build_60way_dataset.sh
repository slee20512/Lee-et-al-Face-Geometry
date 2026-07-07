#!/bin/bash
# Assembles a 60-way (identity x emotion) dataset, with an equal number of
# images per class, from existing render batches on the locker:
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
#
# Uses real file copies (not symlinks/hardlinks): the locker is mounted over
# SMB/CIFS, which doesn't support POSIX symlinks or hardlinks. Copies within
# each class are parallelized (xargs -P) to offset per-file network overhead.
#
# Resumable: a class/split is skipped if the destination already has exactly
# the target count, so a rerun after any failure only redoes the incomplete
# class. Each copy batch is retried once on failure to absorb transient SMB
# errors under sustained concurrent load; note the file-slicing below uses
# `head | tail` (not `tail +K | head`) specifically to avoid a pipefail/SIGPIPE
# false failure when head truncates a much longer upstream stream early.

set -eo pipefail

SRC_BASE="/mnt/smb/locker/issa-locker/users/Seojin/data/face_data/vbsle_50k_texture_colorbg"
OUT_BASE="$SRC_BASE/60way_IDEM_colorbg"
N_TRAIN=1400
N_VAL=350
PARALLEL=8

mkdir -p "$OUT_BASE/train" "$OUT_BASE/val"

# copy_class SRC_DIR DEST_DIR N [SKIP]
# copies the first N files (sorted) from SRC_DIR into DEST_DIR, optionally
# skipping the first SKIP files (used to carve a val split out of a flat pool)
copy_class() {
  local src_dir="$1" dest_dir="$2" n="$3" skip="${4:-0}"
  mkdir -p "$dest_dir"
  local have
  have=$(find "$dest_dir" -maxdepth 1 -type f | wc -l)
  if [ "$have" -eq "$n" ]; then
    echo "  $dest_dir already has $n files, skipping"
    return 0
  fi
  rm -f "$dest_dir"/*
  local filelist
  filelist=$(mktemp)
  find "$src_dir" -maxdepth 1 -type f | sort > "$filelist"
  local attempt
  for attempt in 1 2; do
    if head -n "$((skip + n))" "$filelist" | tail -n "$n" | xargs -P "$PARALLEL" -I{} cp {} "$dest_dir/"; then
      break
    elif [ "$attempt" -eq 2 ]; then
      rm -f "$filelist"
      echo "  FAILED after retry: $dest_dir" >&2
      return 1
    else
      echo "  attempt $attempt failed for $dest_dir, retrying in 10s..." >&2
      sleep 10
    fi
  done
  rm -f "$filelist"
  echo "  $dest_dir done ($(find "$dest_dir" -maxdepth 1 -type f | wc -l) files)"
}

echo "--- 56 existing identity x emotion classes ---"
for split in train val; do
  n=$N_TRAIN
  [ "$split" = "val" ] && n=$N_VAL
  for classdir in "$SRC_BASE/56way_IDEM_colorbg/$split"/*/; do
    cls=$(basename "$classdir")
    copy_class "$classdir" "$OUT_BASE/$split/$cls" "$n"
  done
done

echo "--- elias_neutral, neptune_neutral (already train/val split) ---"
for pair in "elias_neutral:elias" "neptune_neutral:neptune"; do
  cls="${pair%%:*}"
  src_name="${pair##*:}"
  copy_class "$SRC_BASE/texture_colorbg_2way_elias_neptune/train/$src_name" "$OUT_BASE/train/$cls" "$N_TRAIN"
  copy_class "$SRC_BASE/texture_colorbg_2way_elias_neptune/val/$src_name" "$OUT_BASE/val/$cls" "$N_VAL"
done

echo "--- ashley_neutral, josh_neutral (flat pools, splitting train/val ourselves) ---"
for pair in "ashley_neutral:vbsl50k_ashley_neutral" "josh_neutral:vbsl50k_josh_neutral"; do
  cls="${pair%%:*}"
  src_dir="$SRC_BASE/${pair##*:}"
  copy_class "$src_dir" "$OUT_BASE/train/$cls" "$N_TRAIN" 0
  copy_class "$src_dir" "$OUT_BASE/val/$cls" "$N_VAL" "$N_TRAIN"
done

echo "--- verifying per-class counts ---"
for split in train val; do
  for d in "$OUT_BASE/$split"/*/; do
    cnt=$(find "$d" -maxdepth 1 -type f | wc -l)
    echo "$split/$(basename "$d"): $cnt"
  done
done
