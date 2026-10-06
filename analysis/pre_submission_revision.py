"""Reproduce the September 2026 pre-submission tables without training.

Run --check to verify the manuscript tables; --render emits a JSON bundle
of derived artifacts for review. This script never changes the manuscript.
Historical panel confidence intervals in results are not used here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from headline_systems import FEATURED, GRID, load_cells as load_predictions, score, short
from dumps import _auc, _f1_macro
from register_b import (BACKBONES, CNN, TRANSFORMERS, DATASETS, RGB_PRESERVING,
                        CHROMA_ONLY, load_cells, adding_rgb_deltas)
from colour_space_tables import PRETTY

METRICS = ("Accuracy", "Macro-F1", "Sensitivity", "Specificity", "AUC-ROC")
REPETITIONS = 10000
BOOTSTRAP_SEED = 20260906
ROW_END = r" \\"


def metrics(y, p):
    pred = p >= .5
    tp = np.sum(pred & (y == 1))
    tn = np.sum(~pred & (y == 0))
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    m, n = tp + fn, tn + fp
    if not m or not n:
        return None
    auc = (rankdata(p)[y == 1].sum() - m * (m + 1) / 2) / (m * n)
    return np.array([100 * (tp + tn) / len(y),
                     tp / (2 * tp + fp + fn) + tn / (2 * tn + fp + fn),
                     100 * tp / m, 100 * tn / n, auc])


def paired_bootstrap(y, rgb, multi, groups, seed):
    """Non-stratified cluster percentile bootstrap, common draws for both arms.

    All observations in a sampled group travel together; duplicated groups are
    duplicated in full. Models, thresholds, and ensemble composition stay fixed.
    The output estimates case-sampling uncertainty, not training/selection error.
    """
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    samples = []
    discarded = 0
    while len(samples) < REPETITIONS:
        idx = np.concatenate([members[g] for g in rng.integers(len(unique), size=len(unique))])
        a, b = metrics(y[idx], rgb[idx]), metrics(y[idx], multi[idx])
        if a is None or b is None:
            discarded += 1
            continue
        samples.append(np.stack([a, b, b - a]))
    samples = np.asarray(samples)
    a, b = metrics(y, rgb), metrics(y, multi)
    estimate = np.stack([a, b, b - a])
    lo, hi = np.quantile(samples, [.025, .975], axis=0)
    return {
        "n_units": len(y), "n_groups": len(unique), "discarded_draws": discarded,
        "seed": seed, "repetitions": REPETITIONS,
        "rows": [{"metric": metric, "rgb": estimate[0, i], "multi": estimate[1, i],
                  "delta": estimate[2, i], "rgb_ci": [lo[0, i], hi[0, i]],
                  "multi_ci": [lo[1, i], hi[1, i]], "delta_ci": [lo[2, i], hi[2, i]]}
                 for i, metric in enumerate(METRICS)],
    }


def fmt(x, digits=2, signed=False):
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def interval(values, digits=2, signed=False):
    return "[" + ", ".join(fmt(v, digits, signed) for v in values) + "]"


def table(caption, label, spec, header, rows, colsep="4pt"):
    return "\n".join([r"\begin{table*}[htbp]", r"\centering",
                      r"\caption{" + caption + "}", r"\label{" + label + "}",
                      r"\footnotesize", r"\setlength{\tabcolsep}{" + colsep + "}",
                      r"\begin{tabular}{" + spec + "}", r"\toprule", header,
                      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}"])


def build_bundle():
    cells, predictions = load_cells(), load_predictions()
    clinical, selection, featured_rows = {}, [], []
    for ds_index, ds in enumerate(DATASETS):
        bb = FEATURED[ds]
        candidates = sorted((k for k, v in predictions.items()
                             if k[:2] == (ds, bb) and k[2] != "RGB" and "val" in v))
        selected = max(candidates, key=lambda k: _auc(predictions[k]["val"]["true"],
                                                     predictions[k]["val"]["prob"]))
        selection.extend({"dataset": ds, "backbone": bb, "colorset": k[2],
                          "validation_auc": _auc(predictions[k]["val"]["true"], predictions[k]["val"]["prob"]),
                          "selected": k == selected} for k in candidates)
        a, b = predictions[(ds, bb, "RGB")]["test"], predictions[selected]["test"]
        for field in ("unit_ids", "true", "img_paths", "img_patient_ids"):
            np.testing.assert_array_equal(a[field], b[field], err_msg=f"unaligned {ds}/{field}")
        assert a["n_seeds"] == b["n_seeds"] == 5
        groups = a["unit_ids"] if ds == "NeoJaundice" else a["img_patient_ids"]
        expected = (149, 149) if ds == "NeoJaundice" else (152, 151)
        assert (len(a["true"]), len(np.unique(groups))) == expected
        result = paired_bootstrap(a["true"], a["prob"], b["prob"], groups,
                                  BOOTSTRAP_SEED + ds_index)
        result["backbone"], result["colorset"] = bb, selected[2]
        clinical[ds] = result
        unit = "patient" if ds == "NeoJaundice" else "image"
        positives = int(np.sum(a["true"]))
        if featured_rows:
            featured_rows.append(r"\addlinespace[3pt]")
        featured_rows.append(r"\multicolumn{8}{@{}l}{\textit{" + ds + "} --- " + unit + r" level ($N=" +
                             str(expected[0]) + r"$, " + str(positives) + r" positive)}" + ROW_END)
        # Linha da estimativa e, logo abaixo, a do IC 95% em corpo menor, para todas as métricas.
        digits = {"Accuracy": 2, "Macro-F1": 3, "Sensitivity": 1, "Specificity": 1, "AUC-ROC": 3}
        order = ("Accuracy", "Sensitivity", "Specificity", "Macro-F1", "AUC-ROC")
        rows = {r["metric"]: r for r in result["rows"]}
        def sfmt(x, d, signed):
            # Um limite que arredonda para zero sai sem sinal ("0.0", não "+0.0").
            return fmt(0.0, d) if signed and round(x, d) == 0 else fmt(x, d, signed)
        def ci_row(key, delta=False):
            cells = []
            for m in order:
                d = 4 if (delta and m == "AUC-ROC") else digits[m]
                lo, hi = rows[m][key]
                # 6 pt explícitos: nesta classe \scriptsize (9 pt) é maior do que o corpo da tabela (7 pt).
                cells.append(r"{\fontsize{6}{7}\selectfont [" + sfmt(lo, d, delta) + ", " + sfmt(hi, d, delta) + "]}")
            return "   & & " + " & ".join(cells[:3]) + " & & " + " & ".join(cells[3:]) + ROW_END
        fns = {}
        for kind, cs, arm in (("rgb", "RGB", a), ("multi", selected[2], b)):
            s = score(arm, .5)
            fns[kind] = s["fn"]
            values = [PRETTY[bb], short(cs), fmt(s["accuracy"]), fmt(s["sensitivity"], 1),
                      fmt(s["specificity"], 1), str(s["fn"]), fmt(s["f1_macro"], 3), fmt(s["auc_roc"], 3)]
            featured_rows.append("  " + " & ".join(values) + ROW_END)
            featured_rows.append(ci_row(kind + "_ci"))
        delta = [r"\multicolumn{2}{@{}l}{\quad Difference}"]
        for m in order[:3]:
            delta.append(fmt(rows[m]["delta"], digits[m], signed=True))
        delta.append(f"{fns['multi'] - fns['rgb']:+d}")
        delta.append(fmt(rows["Macro-F1"]["delta"], 3, signed=True))
        delta.append(fmt(rows["AUC-ROC"]["delta"], 4, signed=True))
        featured_rows.append("  " + " & ".join(delta) + ROW_END)
        featured_rows.append(ci_row("delta_ci", delta=True))
        if ds == "NeoJaundice":
            img_pos = int(np.sum(a["img_true"]))
            featured_rows.append(r"\addlinespace[3pt]")
            featured_rows.append(r"\multicolumn{8}{@{}l}{\textit{NeoJaundice} --- image level (447 images from 149 infants, " +
                                 str(img_pos) + r" positive; no intervals)}" + ROW_END)
            for cs, arm in (("RGB", a), (selected[2], b)):
                s = score(dict(arm, prob=arm["img_prob"], true=arm["img_true"]), .5)
                featured_rows.append("  " + " & ".join([PRETTY[bb], short(cs), fmt(s["accuracy"]),
                                     fmt(s["sensitivity"], 1), fmt(s["specificity"], 1), str(s["fn"]),
                                     fmt(s["f1_macro"], 3), fmt(s["auc_roc"], 3)]) + ROW_END)
            featured_rows.append(r"\addlinespace[2pt]")
            featured_rows.append(r"\multicolumn{8}{@{}l}{\textit{Published benchmark}~\cite{SkinDataset:2023} --- image level, different partition}" + ROW_END)
            for name, acc, auc in (("EfficientNet-B4", "75.2", "0.829"), ("DenseNet-121", "74.2", "0.814"), ("Swin-base", "71.6", "0.794")):
                featured_rows.append("  " + name + " & n.r. & " + acc + " & n.r. & n.r. & n.r. & n.r. & " + auc + ROW_END)

    blocks = {}
    blocks["H2_melhores_sistemas"] = table(
        r"Illustrative systems in \emph{Register A}: five-seed ensembles at threshold $0.5$. Chromatic arms maximise ensemble validation AUC within the available subset (Section~\ref{sec:best_systems}), not the complete factorial. Below each estimate, in small type, is its 95\% cluster-bootstrap percentile confidence interval (10,000 draws, the same groups resampled for both arms); \emph{Difference} rows give the paired difference, multi-space minus RGB, with its paired interval. FN, false negatives. Image-level NeoJaundice rows, without intervals, provide descriptive context for the published benchmark; n.r., not reported. L, Y and H denote LAB, YCrCb and HSV.",
        "tab:featured_systems", "@{}l l r r r r r r@{}",
        r"\textbf{Backbone} & \textbf{Colour set} & \textbf{Acc. (\%)} & \textbf{Sens. (\%)} & \textbf{Spec. (\%)} & \textbf{FN} & \textbf{Macro-F1} & \textbf{AUC}" + ROW_END,
        featured_rows)
    panel_rows = []
    for ds in DATASETS:
        panel_rows.append(r"\multicolumn{6}{@{}l}{\textbf{" + ds + "}}" + ROW_END)
        for bb in BACKBONES:
            acc = [cells[(ds, bb, cs)].accuracy - cells[(ds, bb, "RGB")].accuracy for cs in RGB_PRESERVING]
            auc = [cells[(ds, bb, cs)].auc - cells[(ds, bb, "RGB")].auc for cs in RGB_PRESERVING]
            panel_rows.append(" & ".join([PRETTY[bb], fmt(st.mean(acc), 2, True),
                              interval([min(acc), max(acc)], 2, True), f"{sum(x > 0 for x in acc)}/7",
                              fmt(st.mean(auc), 4, True), f"{sum(x > 0 for x in auc)}/7"]) + ROW_END)
    blocks["T11_por_backbone_medio"] = table(
        r"Descriptive effects of adding chromatic representations to RGB, in \emph{Register B}. Each entry averages all seven RGB-preserving sets against the same backbone's RGB baseline. Accuracy differences and their observed ranges use percentage points; Pos. counts strictly positive accuracy differences and Pos.\ AUC strictly positive AUC differences among the seven. Mean AUC differences use the original AUC scale. Ranges are not confidence intervals, and these fixed-factorial summaries do not test significance. Not directly comparable with Register A in Table~\ref{tab:featured_systems}.",
        "tab:per_backbone_mean", r"@{}l r c r r r@{}",
        r"\textbf{Backbone} & \textbf{Mean $\Delta$Acc.} & \textbf{Acc. range} & \textbf{Pos.} & \textbf{Mean $\Delta$AUC} & \textbf{Pos.\ AUC}" + ROW_END,
        panel_rows)
    add = adding_rgb_deltas(cells)
    anchor_rows, family_rows, auc_rows = [], [], []
    for kind in ("removing", "adding"):
        title = "Transformed-only inputs versus RGB baseline" if kind == "removing" else "Adding RGB to the same transformed base set"
        anchor_rows.append(r"\multicolumn{5}{@{}l}{\textit{" + title + "}}" + ROW_END)
        for ds in DATASETS:
            d = ([cells[(ds, bb, cs)].accuracy - cells[(ds, bb, "RGB")].accuracy for bb in BACKBONES for cs in CHROMA_ONLY]
                 if kind == "removing" else [add[(ds, bb, cs)] for bb in BACKBONES for cs in RGB_PRESERVING])
            anchor_rows.append("  " + " & ".join([ds, fmt(st.mean(d), 2, True), interval([min(d), max(d)], 2, True),
                                fmt(st.median(d), 2, True), f"{sum(x > 0 for x in d)}/105"]) + ROW_END)
    blocks["T5_rgb_indispensavel"] = table(
        r"RGB removal and addition in \emph{Register B}, with all 105 contrasts per dataset included. Means, observed ranges, medians and positive counts are descriptive summaries of the fixed factorial. Ranges are not confidence intervals. Differences are in percentage points.",
        "tab:rgb_indispensable", "@{}l r c r r@{}",
        r"\textbf{Dataset} & \textbf{Mean $\Delta$Acc.} & \textbf{Range} & \textbf{Median} & \textbf{Positive}" + ROW_END, anchor_rows)
    for ds in DATASETS:
        family_rows.append(r"\multicolumn{7}{@{}l}{\textit{" + ds + "}}" + ROW_END)
        for fam, bbs in (("CNN", CNN), ("Transformer", TRANSFORMERS)):
            values = [fam, str(len(bbs)), fmt(st.mean(cells[(ds, bb, "RGB")].accuracy for bb in bbs))]
            for spaces in (RGB_PRESERVING, CHROMA_ONLY):
                d = [cells[(ds, bb, cs)].accuracy - cells[(ds, bb, "RGB")].accuracy for bb in bbs for cs in spaces]
                values += [fmt(st.mean(d), 3, True), interval([min(d), max(d)], 2, True)]
            family_rows.append("  " + " & ".join(values) + ROW_END)
        for bb in BACKBONES:
            d = [cells[(ds, bb, cs)].auc - cells[(ds, bb, "RGB")].auc for cs in RGB_PRESERVING]
            auc_rows.append(" & ".join([ds, PRETTY[bb], fmt(cells[(ds, bb, "RGB")].auc, 4),
                            fmt(st.mean(d), 4, True), interval([min(d), max(d)], 4, True), f"{sum(x > 0 for x in d)}/7"]) + ROW_END)
    blocks["T9_familia_cenario"] = table(
        r"Architecture-family summaries in \emph{Register B}: 56 CNN and 49 transformer contrasts per operation and dataset. Ranges are observed extrema across cells, not confidence intervals. All quantities describe the evaluated checkpoints. Dataset differences are not controlled contrasts.",
        "tab:family_setting", "@{}l c r r c r c@{}",
        r"& & \textbf{RGB} & \multicolumn{2}{c}{\textbf{Adding to RGB}} & \multicolumn{2}{c}{\textbf{Removing RGB}}" + ROW_END + "\n" +
        r"\textbf{Family} & \textbf{Models} & \textbf{Acc. (\%)} & $\Delta$\textbf{Acc.} & \textbf{Range} & $\Delta$\textbf{Acc.} & \textbf{Range}" + ROW_END, family_rows)
    sup_rows = []
    for ds in DATASETS:
        sup_rows.append(r"\multicolumn{4}{l}{\textbf{" + ds + "}}" + ROW_END)
        for row in clinical[ds]["rows"]:
            digits = 4 if row["metric"] in ("Macro-F1", "AUC-ROC") else 2
            values = [row["metric"]]
            for kind in ("rgb", "multi", "delta"):
                values.append(fmt(row[kind], digits, kind == "delta") + " " + interval(row[kind + "_ci"], digits, kind == "delta"))
            sup_rows.append(" & ".join(values) + ROW_END)
    supplement = "\n".join([
        r"\documentclass[10pt]{article}", r"\usepackage[a4paper,margin=18mm]{geometry}",
        r"\usepackage{booktabs,amsmath,float}", r"\renewcommand{\thetable}{S\arabic{table}}",
        r"\begin{document}", r"\section*{Supplementary statistical results}",
        "Companion to the pre-submission manuscript on multi-space chromatic early fusion. "
        "All predictions, thresholds and models are fixed. The test set is used only for evaluation.",
        table("Available validation candidates for the two illustrative systems. Selection is restricted to preserved prediction files; it is not an ensemble-AUC search across all 14 alternatives.",
              "sup:selection", "@{}l l l r l@{}", "Dataset & Backbone & Colour set & Val. AUC & Selected" + ROW_END,
              [" & ".join([r["dataset"], PRETTY[r["backbone"]], short(r["colorset"]), fmt(r["validation_auc"], 4), "yes" if r["selected"] else "no"]) + ROW_END for r in selection]),
        table("Fixed-system performance and paired differences (multi-space minus RGB), with 95\\% cluster-bootstrap percentile intervals from 10,000 draws. Accuracy, sensitivity, specificity and their differences use percentage units; F1 and AUC use their original scales.",
              "sup:clinical", "@{}l c c c@{}", "Metric & RGB [95\\% CI] & Multi-space [95\\% CI] & Difference [95\\% CI]" + ROW_END, sup_rows),
        "NJN retains image-level scoring and resamples all 151 inferred test groups, carrying all images of each sampled group together (152 images in total). "
        "NeoJaundice first averages probabilities per patient and then resamples its 149 patients. Sampling is non-stratified, with replacement. "
        "Identical draws are used for both arms. Single-class draws, if any, are discarded. Random seeds are 20260906 and 20260907, respectively. "
        "Intervals are conditional on the fitted ensembles, the selected configurations and the frozen split. They do not include training, selection or cross-domain uncertainty. "
        "Paired intervals are unadjusted descriptive uncertainty estimates, not a family of confirmatory significance tests.",
        r"\clearpage",
        table("Threshold-independent companion to the accuracy panels: all seven RGB-preserving alternatives versus each backbone's RGB baseline in Register B. Observed ranges describe treatment variation, not sampling uncertainty. No configuration is selected.",
              "sup:auc", "@{}l l r r c r@{}", "Dataset & Backbone & RGB AUC & Mean $\\Delta$AUC & Range & Positive" + ROW_END, auc_rows),
        r"\end{document}", ""]).replace(r"\begin{table*}[htbp]", r"\begin{table}[H]").replace(r"\end{table*}", r"\end{table}")
    attribution_rows = []
    with (ROOT / "results/G2_attribution_shift.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            method = "Grad-CAM" if row["family"] == "cnn" else "Rollout"
            attribution_rows.append(" & ".join([row["dataset"], PRETTY[row["backbone"]], short(row["colorset"]),
                                    method + " " + row["grid"].replace("x", r"$\times$"),
                                    fmt(float(row["r_within_rgb"]), 3), fmt(float(row["r_within_multi"]), 3),
                                    fmt(float(row["r_between"]), 3)]) + ROW_END)
    blocks["G2_atribuicao"] = table(
        r"Attribution agreement on the six retrained pairs (Section~\ref{sec:attribution}). Each value is the median over test images of the per-image Pearson correlation between two attribution maps: within-arm columns average the 10 seed pairs of the RGB and of the multi-space arm, the between-arm column the 25 cross-arm seed pairs. A constant Grad-CAM map has no defined correlation: such seed pairs are skipped, and test images left without a defined value (one for MobileNetV3-L, two for DenseNet121) are omitted from the medians. Maps come from retrained replicas, not from the campaign checkpoints. L, Y and H denote LAB, YCrCb and HSV.",
        "tab:attribution", "@{}l l l l r r r@{}",
        r"\textbf{Dataset} & \textbf{Backbone} & \textbf{Colour set} & \textbf{Method, grid} & \textbf{Within RGB} & \textbf{Within multi} & \textbf{Between arms}" + ROW_END,
        attribution_rows)

    # Factos em prosa, recalculados dos artefactos, que o --check exige encontrar no .tex.
    winners = [json.loads(Path(f).read_text())["best_params"]
               for f in sorted(glob.glob(str(ROOT / "evidence/v7/best_params/*.json")))]
    assert len(winners) == 30
    unfreeze = {k: sum(w["n_unfreeze"] == k for w in winners) for k in (-1, 100, 50, 10, 0)}
    strengths = [w["da_strength"] for w in winners if w["augment"]]
    seed_std = [c.std_seed_auc for c in cells.values()]
    audit = json.loads((ROOT / "results/fusion_block_audit.json").read_text())
    levers = {}
    with (ROOT / "results/H0_alavancas.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            levers[(row["dataset"], row["lever"])] = (int(row["n_cells"]), float(row["vs_manuscript"]))
    positives = {}
    for ds, path, group in (("NJN", "splits/njn_split.csv", "pseudo_patient"), ("NeoJaundice", "splits/neojaundice_split.csv", "patient_id")):
        with (ROOT / path).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                positives[(ds, row["split"])] = positives.get((ds, row["split"]), 0) + int(row["label"])
    val_thr = {}
    for ds in DATASETS:
        bb = FEATURED[ds]
        chosen = next(r["colorset"] for r in selection if r["dataset"] == ds and r["selected"])
        for cs in ("RGB", chosen):
            v = predictions[(ds, bb, cs)]
            best, thr = -1.0, 0.5
            for cand in GRID:
                f1 = _f1_macro(v["val"]["true"], (v["val"]["prob"] >= cand).astype(int))
                if f1 > best + 1e-12 or (abs(f1 - best) <= 1e-12 and abs(cand - 0.5) < abs(thr - 0.5)):
                    best, thr = f1, float(cand)
            val_thr[(ds, cs == "RGB")] = dict(score(v["test"], thr), threshold=thr)
    vt = val_thr
    facts = {
        "val_threshold": (f"on NJN the thresholds become ${vt[('NJN', True)]['threshold']:.2f}$ (RGB) and ${vt[('NJN', False)]['threshold']:.2f}$ (multi-space) "
                          f"and accuracy ${vt[('NJN', True)]['accuracy']:.2f}\\%$ against ${vt[('NJN', False)]['accuracy']:.2f}\\%$, with sensitivity "
                          f"${vt[('NJN', True)]['sensitivity']:.1f}\\%$ against ${vt[('NJN', False)]['sensitivity']:.1f}\\%$; on NeoJaundice the RGB threshold "
                          f"{'stays at' if vt[('NeoJaundice', True)]['threshold'] == 0.5 else 'becomes'} ${vt[('NeoJaundice', True)]['threshold']:.2f}$ and the multi-space one becomes "
                          f"${vt[('NeoJaundice', False)]['threshold']:.2f}$, giving ${vt[('NeoJaundice', True)]['accuracy']:.2f}\\%$ against ${vt[('NeoJaundice', False)]['accuracy']:.2f}\\%$ accuracy "
                          f"and ${vt[('NeoJaundice', True)]['sensitivity']:.1f}\\%$ against ${vt[('NeoJaundice', False)]['sensitivity']:.1f}\\%$ sensitivity "
                          f"({vt[('NeoJaundice', False)]['fn']} false negatives)"),
        "auc_panels": (f"area under the receiver operating characteristic curve (AUC) ranged from ${min(st.mean(cells[(ds, bb, cs)].auc - cells[(ds, bb, 'RGB')].auc for cs in RGB_PRESERVING) for ds in DATASETS for bb in BACKBONES):+.4f}$ "
                       f"to ${max(st.mean(cells[(ds, bb, cs)].auc - cells[(ds, bb, 'RGB')].auc for cs in RGB_PRESERVING) for ds in DATASETS for bb in BACKBONES):+.4f}$"),
        "hpo_winners": (f"learning rates span ${winners_lr_min(winners)}$ to ${winners_lr_max(winners)}$; "
                        f"Adam was chosen in {sum(w['optimizer'] == 'adam' for w in winners)} and Adamax in "
                        f"{sum(w['optimizer'] == 'adamax' for w in winners)}; the backbone was fully unfrozen in {unfreeze[-1]}, "
                        f"unfrozen over its last 50 tensors in {unfreeze[50]}, 100 in {unfreeze[100]} and 10 in {unfreeze[10]}; "
                        f"augmentation was enabled in {len(strengths)}, with strengths from ${min(strengths):.2f}$ to ${max(strengths):.2f}$"),
        "seed_dispersion": f"median standard deviation of the test AUC across the five seeds of a cell is ${st.median(seed_std):.3f}$",
        "fusion_parity": (f"maximum absolute difference of ${audit['parity']['4']['max_abs_diff_vs_rgb_arm'] * 1e3:.1f} \\times 10^{{-3}}$ "
                          f"(mean ${audit['parity']['4']['mean_abs_diff_vs_rgb_arm'] * 1e4:.1f} \\times 10^{{-4}}$)"),
        "fusion_params": f"between {audit['params']['1']['total']} ($K=1$) and {audit['params']['4']['total']} ($K=4$) trainable parameters",
        "levers": (f"over the {levers[('NJN', '+ 5-seed ensemble')][0]} cells per dataset with preserved test predictions, all read at threshold $0.5$, "
                   f"the five-seed ensemble adds $+{levers[('NJN', '+ 5-seed ensemble')][1]:.2f}$ percentage points (p.p.) of accuracy on NJN and "
                   f"$+{levers[('NeoJaundice', '+ 5-seed ensemble')][1]:.2f}$ on NeoJaundice to the per-seed image-level mean, and the patient unit a further "
                   f"$+{levers[('NeoJaundice', '+ protocol unit')][1] - levers[('NeoJaundice', '+ 5-seed ensemble')][1]:.2f}$ p.p.\\ on NeoJaundice"),
        "positives_njn": f"532 / 529 ({positives[('NJN', 'train')]}) & 76 / 75 ({positives[('NJN', 'val')]}) & 152 / 151 ({positives[('NJN', 'test')]})",
        "positives_neo_valtest": f"225 / 75 ({positives[('NeoJaundice', 'val')]}) & 447 / 149 ({positives[('NeoJaundice', 'test')]})",
    }
    return {"blocks": blocks, "clinical": clinical, "selection": selection,
            "canonical_prediction_cells": len(predictions), "supplement": supplement, "facts": facts}


def _sci(x):
    exp = int(np.floor(np.log10(x)))
    return f"{x / 10 ** exp:.1f} \\times 10^{{{exp}}}"


def winners_lr_min(winners):
    return _sci(min(w["lr"] for w in winners))


def winners_lr_max(winners):
    return _sci(max(w["lr"] for w in winners))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true",
                        help="grava results/pre_submission_statistics.json (os blocos LaTeX só existem inline em paper.tex; use --render para os ver)")
    args = parser.parse_args()
    bundle = build_bundle()
    if args.render:
        print(json.dumps(bundle, indent=2))
    if args.write:
        (ROOT / "results/pre_submission_statistics.json").write_text(
            json.dumps({k: v for k, v in bundle.items() if k not in ("blocks", "supplement")}, indent=2))
        print("escrito o JSON de estatísticas; os blocos LaTeX vivem só em paper.tex (--render para os ver)")
    if args.check:
        saved = json.loads((ROOT / "results/pre_submission_statistics.json").read_text())
        assert saved == {k: v for k, v in bundle.items() if k not in ("blocks", "supplement")}
        paper_path = ROOT / "paper/paper.tex"
        if not paper_path.is_file():
            # Repositório público sem o manuscrito: a reprodução numérica continua valendo.
            print("PASS: paired bootstrap, selection artifacts and prose facts reproduced from "
                  "preserved predictions; manuscript comparison skipped (paper/ not present)")
            return
        paper = paper_path.read_text()
        for name, block in bundle["blocks"].items():
            start, end = f"%% <<<TABLE:{name}>>>", f"%% <<<END TABLE:{name}>>>"
            actual = paper.split(start, 1)[1].split(end, 1)[0].strip()
            assert actual == block, f"manuscript table differs: {name}"
        assert "%% <<<APPENDIX:statistics>>>" not in paper
        assert "Supplementary Table" not in paper and "\\ref{app:" not in paper
        for name, text in bundle["facts"].items():
            assert text in paper, f"prose fact not found in manuscript: {name}: {text}"
        print(f"PASS: {len(bundle['blocks'])} integrated manuscript tables, {len(bundle['facts'])} prose facts, "
              "paired bootstrap and selection artifacts")


if __name__ == "__main__":
    main()
