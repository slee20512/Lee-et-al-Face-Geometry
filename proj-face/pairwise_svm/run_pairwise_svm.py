"""Background pairwise-decoding (SVM) runner with a live HTML dashboard.

Ports the `# (N) exp_names_list_*` SVM cells of MyMain_SL.ipynb into a standalone
script: for every task x model x identity/emotion pair it fits a LinearSVC with
100-rep 2-fold stratified CV on the precomputed feature .pth files, and after
each unit rewrites results.json + dashboard.html so you can watch progress in a
browser.

Typical use (on a box with the issa-locker mounted):

    cd proj-face/pairwise_svm
    nohup python run_pairwise_svm.py \
        --tasks CTFR GFR GTFR \
        --models SL_resnet50_finetune_28way_IDEM_combined_seed777_model_best \
                 off_the_shelf_barlowtwins_IDEM_28way_epochs50 \
        --outdir ./results_28way_IDEM_combined \
        > run.out 2>&1 &

    # then open results_28way_IDEM_combined/dashboard.html in a browser
    # (it self-refreshes every 10 s)

The run is resumable: re-launching with the same --outdir skips units already in
results.json. Pass --redo to recompute everything, or --redo-errors to retry only
the failed units.
"""
import argparse
import datetime as dt
import html
import json
import os
import socket
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))  # for identity_map / project imports

from pairwise_tasks import TASKS  # noqa: E402

try:
    from identity_map import resolve_ids  # local-only, gitignored
except ImportError:
    def resolve_ids(path):
        return path


def now_iso():
    return dt.datetime.now().replace(microsecond=0).isoformat()


def fmt_dur(secs):
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# --------------------------------------------------------------------------- #
# core numeric routine -- mirrors the notebook SVM cell exactly
# --------------------------------------------------------------------------- #
def _load_feats(path):
    import torch
    obj = torch.load(resolve_ids(path), map_location="cpu")
    if hasattr(obj, "reshape") and not isinstance(obj, np.ndarray):
        obj = obj.reshape(len(obj), -1)
        return obj.float().cpu().numpy()
    obj = np.asarray(obj)
    return obj.reshape(len(obj), -1)


def decode_pair(path_1, path_2, num_rep, max_iter, rebalance_seed):
    """Return dict with mean/std accuracy, confusion matrix, per-image acc/dist."""
    import torch
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix
    from sklearn import svm

    input_1 = _load_feats(path_1)
    input_2 = _load_feats(path_2)

    n1_raw, n2_raw = len(input_1), len(input_2)

    # rebalance: shuffle the larger class and truncate to the smaller (seeded)
    torch.manual_seed(rebalance_seed)
    if len(input_1) != len(input_2):
        if len(input_2) > len(input_1):
            input_1, input_2 = input_2, input_1
        idx = torch.randperm(input_1.shape[0]).numpy()
        input_1 = input_1[idx][:len(input_2)]

    output_1 = np.ones(len(input_1))
    output_2 = np.zeros(len(input_2))
    n_per_class = len(output_1)

    dists = np.full((2, num_rep, n_per_class), np.nan)
    acc = np.full((2, num_rep, n_per_class), np.nan)
    all_scores = []
    conf_mat = np.zeros((2, 2))

    for rep_index in range(num_rep):
        cv = StratifiedKFold(n_splits=2, shuffle=True)
        scores = []
        for train_index, test_index in cv.split(input_1, output_1):
            X_train = np.concatenate((input_1[train_index], input_2[train_index]), axis=0)
            X_test = np.concatenate((input_1[test_index], input_2[test_index]), axis=0)
            y_train = np.concatenate((output_1[train_index], output_2[train_index]), axis=0)
            y_test = np.concatenate((output_1[test_index], output_2[test_index]), axis=0)

            clf = svm.LinearSVC(penalty="l2", loss="hinge", dual=True, tol=1e-4,
                                fit_intercept=True, C=1.0, max_iter=max_iter)
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)

            scores.append((y_predict == y_test).sum() / len(y_predict))
            conf_mat += confusion_matrix(y_test, y_predict, labels=[0, 1])

            _acc_per_img = (y_predict == y_test).astype("float32")
            n_test = len(test_index)
            acc[0][rep_index][test_index] = _acc_per_img[:n_test]
            acc[1][rep_index][test_index] = _acc_per_img[n_test:]

            dists[0][rep_index][test_index] = clf.decision_function(input_1[test_index])
            dists[1][rep_index][test_index] = clf.decision_function(input_2[test_index]) * -1
        all_scores.append(float(np.mean(scores)))

    all_scores = np.array(all_scores)
    return {
        "mean": float(all_scores.mean()),
        "std": float(all_scores.std()),
        "n1_raw": int(n1_raw),
        "n2_raw": int(n2_raw),
        "n_per_class": int(n_per_class),
        "conf_mat": (conf_mat / num_rep / 2).tolist(),
        "acc_per_img": acc.mean(1),   # (2, n_per_class)   -- for optional npz dump
        "dist": (dists / num_rep),     # (2, num_rep, n_per_class)
    }


# --------------------------------------------------------------------------- #
# task expansion
# --------------------------------------------------------------------------- #
def task_pairs(task):
    cfg = TASKS[task]
    exp = cfg["exp_name_list"]
    if "data_root" in cfg:
        roots = [cfg["data_root"]] * len(exp)
    else:
        roots = cfg["data_root_list"]
    return [
        {"root": roots[i], "exp0": exp[i][0], "exp1": exp[i][1]}
        for i in range(len(exp))
    ]


def feat_path(root, exp, model, postfix):
    return os.path.join(root, f"{exp}_{model}{postfix}.pth")


# --------------------------------------------------------------------------- #
# state I/O
# --------------------------------------------------------------------------- #
def atomic_write(path, text):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #
def acc_cell_style(v):
    # 0.5 -> white, 1.0 -> green
    t = max(0.0, min(1.0, (v - 0.5) / 0.5))
    g = int(255 - 90 * t)
    return f"background:rgb({g},255,{g})" if v >= 0.5 else "background:rgb(255,230,230)"


def render_dashboard(state, refresh):
    meta = state["meta"]
    total = meta["units_total"]
    done = meta["units_done"]
    err = meta["units_error"]
    pct = 100 * done / total if total else 0
    elapsed = time.time() - meta["_start_epoch"]
    rate = meta["compute_secs"] / max(done, 1)
    eta = rate * (total - done)

    p = []
    p.append("<!doctype html><html><head><meta charset='utf-8'>")
    p.append(f"<meta http-equiv='refresh' content='{refresh}'>")
    p.append("<title>pairwise SVM decoding</title>")
    p.append("<style>"
             "body{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#222}"
             "h2{margin:28px 0 6px}table{border-collapse:collapse;margin:6px 0}"
             "td,th{border:1px solid #ccc;padding:2px 6px;text-align:center;white-space:nowrap}"
             "th{background:#f4f4f4}.mdl{text-align:left;max-width:420px;overflow:hidden;text-overflow:ellipsis}"
             ".bar{height:18px;background:#e6e6e6;border-radius:3px;overflow:hidden;width:420px;display:inline-block;vertical-align:middle}"
             ".bar>span{display:block;height:100%;background:#3b82f6}"
             ".wrap{overflow-x:auto;max-width:100%}pre{background:#f7f7f7;padding:10px;overflow-x:auto}"
             ".err{color:#b00}.muted{color:#888}</style></head><body>")

    status = meta["status"]
    p.append(f"<h1>pairwise SVM decoding &mdash; <span class='{'err' if status=='error' else ''}'>{status}</span></h1>")
    p.append("<div class='bar'><span style='width:%.1f%%'></span></div> " % pct)
    p.append(f"<b>{done}/{total}</b> units ({pct:.1f}%)")
    if err:
        p.append(f" &nbsp;<span class='err'>{err} errors</span>")
    p.append("<p class='muted'>")
    p.append(f"started {meta['started']} &middot; updated {meta['updated']} &middot; host {meta['hostname']}<br>")
    p.append(f"elapsed {fmt_dur(elapsed)} &middot; eta {fmt_dur(eta) if done else '?'} "
             f"&middot; ~{fmt_dur(rate)}/unit &middot; num_rep={meta['num_rep']}<br>")
    p.append("models: " + " , ".join(html.escape(m) for m in meta["models"]))
    if meta.get("current"):
        p.append(f"<br>current: {html.escape(meta['current'])}")
    p.append("</p>")

    for task, tdata in state["tasks"].items():
        pairs = tdata["pairs"]
        np_ = len(pairs)
        res = tdata["results"]
        tdone = sum(1 for m in res for k in res[m] if "mean" in res[m][k])
        p.append(f"<h2>{task} <span class='muted'>(nb #{tdata['notebook_id']}, "
                 f"{np_} pairs, {tdone}/{np_ * len(meta['models'])} done)</span></h2>")
        p.append("<div class='wrap'><table>")
        p.append("<tr><th>model</th>" + "".join(
            f"<th title='{html.escape(pairs[i][0])} vs {html.escape(pairs[i][1])}'>{i}</th>"
            for i in range(np_)) + "<th>mean&plusmn;sd</th></tr>")
        for m in meta["models"]:
            row = [f"<tr><td class='mdl' title='{html.escape(m)}'>{html.escape(m)}</td>"]
            vals = []
            for i in range(np_):
                cell = res.get(m, {}).get(str(i))
                if cell is None:
                    row.append("<td class='muted'>&middot;</td>")
                elif "error" in cell:
                    row.append("<td class='err' title='%s'>ERR</td>" % html.escape(cell["error"][:300]))
                else:
                    vals.append(cell["mean"])
                    row.append(f"<td style='{acc_cell_style(cell['mean'])}'>{cell['mean']:.3f}</td>")
            if vals:
                row.append(f"<td><b>{np.mean(vals):.3f}</b>&plusmn;{np.std(vals):.3f}</td>")
            else:
                row.append("<td class='muted'>&middot;</td>")
            row.append("</tr>")
            p.append("".join(row))
        p.append("</table></div>")

        # pastable rows (matches print_pairwise_rows in the notebook)
        lines = []
        for m in meta["models"]:
            cells = []
            for i in range(np_):
                c = res.get(m, {}).get(str(i))
                cells.append(f"{c['mean']:.6f}+-{c['std']:.6f}" if c and "mean" in c else "NA")
            lines.append(m + "\t" + "\t".join(cells))
        p.append("<pre>" + html.escape("\n".join(lines)) + "</pre>")

    # errors
    errs = [(t, m, k, state["tasks"][t]["results"][m][k]["error"])
            for t in state["tasks"] for m in state["tasks"][t]["results"]
            for k in state["tasks"][t]["results"][m]
            if "error" in state["tasks"][t]["results"][m][k]]
    if errs:
        p.append("<h2 class='err'>errors</h2><pre>")
        for t, m, k, e in errs:
            p.append(html.escape(f"[{t}] {m} pair {k}: {e}") + "\n")
        p.append("</pre>")

    p.append("</body></html>")
    return "".join(p)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=sorted(TASKS),
                    help="task names from pairwise_tasks.py (default: all)")
    ap.add_argument("--models", nargs="+", help="model_name stems (required unless --list)")
    ap.add_argument("--outdir", default=os.path.join(os.getcwd(), "pairwise_svm_results"))
    ap.add_argument("--num-rep", type=int, default=100)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--filename-postfix", default="")
    ap.add_argument("--rebalance-seed", type=int, default=7)
    ap.add_argument("--refresh", type=int, default=10, help="dashboard auto-refresh seconds")
    ap.add_argument("--redo", action="store_true", help="recompute every unit")
    ap.add_argument("--redo-errors", action="store_true", help="retry only failed units")
    ap.add_argument("--save-arrays", action="store_true",
                    help="also dump per-image acc/dist to <outdir>/arrays/*.npz")
    ap.add_argument("--list", action="store_true", help="list available tasks and exit")
    args = ap.parse_args()

    if args.list:
        for k in sorted(TASKS):
            c = TASKS[k]
            print(f"{k:32s} nb#{c['notebook_id']:<2}  {len(c['exp_name_list'])} pairs")
        return

    if not args.models:
        ap.error("--models is required")
    unknown = [t for t in args.tasks if t not in TASKS]
    if unknown:
        ap.error(f"unknown tasks: {unknown}\navailable: {sorted(TASKS)}")

    os.makedirs(args.outdir, exist_ok=True)
    if args.save_arrays:
        os.makedirs(os.path.join(args.outdir, "arrays"), exist_ok=True)
    results_path = os.path.join(args.outdir, "results.json")
    dash_path = os.path.join(args.outdir, "dashboard.html")

    prev = load_state(results_path)

    # build / merge state
    state = {"meta": {}, "tasks": {}}
    for task in args.tasks:
        pairs = [[p["exp0"], p["exp1"]] for p in task_pairs(task)]
        state["tasks"][task] = {
            "notebook_id": TASKS[task]["notebook_id"],
            "source_name": TASKS[task]["source_name"],
            "pairs": pairs,
            "results": {m: {} for m in args.models},
        }
    if prev and not args.redo:
        for task, td in prev.get("tasks", {}).items():
            if task not in state["tasks"]:
                continue
            for m, cells in td.get("results", {}).items():
                if m not in state["tasks"][task]["results"]:
                    continue
                for k, cell in cells.items():
                    if "error" in cell and args.redo_errors:
                        continue
                    state["tasks"][task]["results"][m][k] = cell

    # unit list
    units = []
    for task in args.tasks:
        tp = task_pairs(task)
        for m in args.models:
            for i in range(len(tp)):
                if str(i) in state["tasks"][task]["results"][m]:
                    continue
                units.append((task, m, i, tp[i]))

    total_units = sum(len(task_pairs(t)) for t in args.tasks) * len(args.models)
    start_epoch = time.time()
    prev_compute = (prev or {}).get("meta", {}).get("compute_secs", 0.0) if not args.redo else 0.0

    state["meta"] = {
        "status": "running",
        "started": (prev or {}).get("meta", {}).get("started", now_iso()) if not args.redo else now_iso(),
        "updated": now_iso(),
        "hostname": socket.gethostname(),
        "models": args.models,
        "tasks": args.tasks,
        "num_rep": args.num_rep,
        "units_total": total_units,
        "units_done": total_units - len(units),
        "units_error": sum(1 for t in state["tasks"] for m in state["tasks"][t]["results"]
                           for k in state["tasks"][t]["results"][m]
                           if "error" in state["tasks"][t]["results"][m][k]),
        "compute_secs": prev_compute,
        "current": None,
        "_start_epoch": start_epoch,
    }

    def flush():
        state["meta"]["updated"] = now_iso()
        atomic_write(results_path, json.dumps(state, indent=1, default=_json_default))
        atomic_write(dash_path, render_dashboard(state, args.refresh))

    print(f"{now_iso()}  {len(units)} units to run  ({state['meta']['units_done']}/{total_units} already done)")
    print(f"{now_iso()}  dashboard: {dash_path}")
    flush()

    for n, (task, model, i, pair) in enumerate(units, 1):
        tag = f"{task} | {model} | pair {i} ({pair['exp0']} vs {pair['exp1']})"
        state["meta"]["current"] = tag
        print(f"{now_iso()}  [{n}/{len(units)}] {tag}", flush=True)
        t0 = time.time()
        try:
            out = decode_pair(
                feat_path(pair["root"], pair["exp0"], model, args.filename_postfix),
                feat_path(pair["root"], pair["exp1"], model, args.filename_postfix),
                args.num_rep, args.max_iter, args.rebalance_seed,
            )
            secs = time.time() - t0
            if args.save_arrays:
                npz = os.path.join(args.outdir, "arrays", f"{task}__{model}__pair{i:03d}.npz")
                np.savez_compressed(npz, acc_per_img=out.pop("acc_per_img"), dist=out.pop("dist"))
                out["arrays_npz"] = os.path.relpath(npz, args.outdir).replace(os.sep, "/")
            else:
                out.pop("acc_per_img"); out.pop("dist")
            out["secs"] = round(secs, 1)
            out["done_at"] = now_iso()
            state["tasks"][task]["results"][model][str(i)] = out
            state["meta"]["units_done"] += 1
            state["meta"]["compute_secs"] += secs
            print(f"{now_iso()}      acc={out['mean']:.4f}+-{out['std']:.4f}  ({fmt_dur(secs)})", flush=True)
        except Exception:
            tb = traceback.format_exc()
            state["tasks"][task]["results"][model][str(i)] = {"error": tb.strip().splitlines()[-1], "done_at": now_iso()}
            state["meta"]["units_error"] += 1
            print(f"{now_iso()}      ERROR\n{tb}", flush=True)
        flush()

    state["meta"]["status"] = "done" if state["meta"]["units_error"] == 0 else "done-with-errors"
    state["meta"]["current"] = None
    flush()
    print(f"{now_iso()}  finished: {state['meta']['units_done']}/{total_units} done, "
          f"{state['meta']['units_error']} errors")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
