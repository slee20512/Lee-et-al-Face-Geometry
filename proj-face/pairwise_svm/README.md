# pairwise_svm — background SVM pairwise-decoding with a live HTML dashboard

Standalone port of the `# (N) exp_names_list_*` SVM cells in `MyMain_SL.ipynb`.
Runs the same LinearSVC + 100-rep 2-fold stratified CV over the precomputed
feature `.pth` files, one task × model × pair at a time, and after **every unit**
rewrites `results.json` + `dashboard.html` so you can watch a long run in a
browser.

## Files

| file | what |
|---|---|
| `pairwise_tasks.py` | auto-generated task table (data_root + exp_name pairs for each `# (N)` block) |
| `gen_pairwise_tasks.py` | regenerates `pairwise_tasks.py` from the notebook |
| `run_pairwise_svm.py` | the job |

## Run it

On a machine with the issa-locker mounted, in the conda env that has torch +
scikit-learn:

```bash
cd proj-face/pairwise_svm

nohup python run_pairwise_svm.py \
    --tasks CTFR GFR GTFR \
    --models SL_resnet50_finetune_28way_IDEM_combined_seed777_model_best \
             off_the_shelf_barlowtwins_IDEM_28way_epochs50 \
    --outdir ./results_28way_IDEM_combined \
    > results_28way_IDEM_combined/run.out 2>&1 &
```

Then open `results_28way_IDEM_combined/dashboard.html` in a browser. It
self-refreshes every 10 s (`--refresh N` to change). `run.out` has the same
progress as plain text.

Omit `--tasks` to run all 17. `python run_pairwise_svm.py --list` prints them.

## Output

`<outdir>/`
- `results.json` — full results: per-pair `mean`, `std`, `conf_mat`, raw class
  sizes, timing. Also `meta` with progress / ETA.
- `dashboard.html` — progress bar, one colour-coded table per task (rows =
  models, columns = pairs, last column = task mean ± sd), an errors section, and
  a **pastable-rows** `<pre>` block per task matching the notebook's
  `print_pairwise_rows` output.
- `pastable_rows.txt` — the same pastable rows, all tasks, as plain text; copy
  straight back into the notebook.
- `arrays/<task>__<model>__pairNNN.npz` — only with `--save-arrays`: per-image
  `acc_per_img` (2, n) and `dist` (2, num_rep, n) for downstream i1 analyses.

## Resuming

Re-launching with the same `--outdir` skips units already in `results.json`.
- `--redo-errors` retries only the failed units (e.g. after a missing feature
  file is generated).
- `--redo` recomputes everything.

## Notes

- CPU only, single process, one pair at a time. A 66-pair task × 2 models is
  ~132 fits × 100 reps × 2 folds; wall time depends on feature dim and image
  count. Run several `--outdir`s in parallel for different task groups if needed.
- `StratifiedKFold` is unseeded (matches the notebook), so per-pair numbers move
  by ~1 std between runs.
- If `identity_map.py` is present its `resolve_ids` is used for path rewriting,
  same as the notebook.
- Regenerate the task table after editing the notebook's `exp_name_list`s:
  `python gen_pairwise_tasks.py`.
