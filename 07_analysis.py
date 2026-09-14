#!/usr/bin/env python3
"""
07_analysis.py
==============
Week 13-14 deliverable: comparative statistical analysis across datasets and
interaction-volume bands.

Consumes the pipeline from 03-06. Performs one scoring pass to obtain
per-user metric vectors, caches them, and runs all statistics from the cache
so the analysis can be re-run cheaply while the write-up is iterated.

What is tested, and why in this form
------------------------------------

1. OMNIBUS, WITHIN EACH DATASET x BAND
   Friedman test across all models, blocked by user. Every model scores every
   sampled user, so the blocks are complete and the paired structure is used
   rather than discarded. A significant Friedman result licenses post-hocs;
   without it, pairwise comparisons within that cell are not interpreted.

2. POST-HOC PAIRWISE
   Wilcoxon signed-rank for each model pair within each dataset x band cell,
   with matched-pairs rank-biserial correlation as the effect size. Raw
   p-values are corrected two ways: Holm within each cell (family-wise, the
   conservative reading) and Benjamini-Hochberg across the entire set of
   comparisons (false-discovery, the reading appropriate to an exploratory
   band sweep). Both columns are reported; the write-up should state which
   it relies on.

3. THE DIRECTIONAL HYPOTHESIS
   The design predicts that the hybrid's advantage is largest where history
   is scarcest and decays as volume accumulates. That is an ordered
   alternative, not five unrelated comparisons, and testing it as five
   comparisons both loses power and invites a multiple-testing objection.
   Two complementary tests are used:
     - Jonckheere-Terpstra for a monotone trend in the per-user advantage
       across the ordered bands.
     - OLS of the per-user advantage on log1p(interaction count), with HC1
       heteroskedasticity-robust standard errors. A reliably negative slope
       is the quantitative form of the claim.

4. DIFFERENCE-IN-DIFFERENCES ACROSS DATASETS
   Whether the cold-start advantage is larger on the sparser dataset is
   tested as a difference-in-differences: (advantage in cold bands) minus
   (advantage in warm bands), computed per dataset, then differenced across
   datasets. The interval comes from a user-level bootstrap, because the two
   datasets have different user populations and no paired test applies.

5. CROSS-DATASET CONSISTENCY
   Kendall tau and Spearman rho between the model orderings obtained on each
   dataset, per band. High concordance means the findings generalise; low
   concordance in cold bands specifically would be a substantive finding
   about dataset dependence, not a nuisance.

6. ACCURACY-COVERAGE TRADE-OFF
   Correlation between accuracy gain and cold-item coverage loss across
   models, since a hybrid that wins precision by routing away from cold
   items has not solved the problem it claims to.

Usage
-----
    python 07_analysis.py --processed ./data/processed
    python 07_analysis.py --processed ./data/processed --from-cache
    python 07_analysis.py --processed ./data/processed --treatment hybrid_shrinkage \\
        --control cf_only --metric ndcg@10

Outputs (under --out/analysis):
    per_user_metrics.parquet    cached per-user vectors, all datasets/models
    descriptives.csv            dataset x band x model means with intervals
    omnibus_friedman.csv
    pairwise_posthoc.csv        Holm and BH corrected, with effect sizes
    volume_trend.csv            Jonckheere-Terpstra and robust OLS
    cross_dataset.csv           rank concordance per band
    diff_in_diff.csv
    tradeoff.csv
    figures/*.png
    analysis_summary.md         narrative summary with the numbers filled in
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BANDS_ORDER = [
    "Band 0 (0)", "Band 1 (1-4)", "Band 2 (5-19)",
    "Band 3 (20-49)", "Band 4 (50+)",
]
COLD_BANDS = ["Band 0 (0)", "Band 1 (1-4)"]
K_VALUES = (5, 10, 20)
RNG_SEED = 42

_T0 = time.time()


def log(msg: str, indent: int = 0) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {' ' * indent}{msg}", flush=True)


def load_module(path: Path, name: str):
    if not path.exists():
        sys.exit(f"missing {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def band_of(n: int) -> str:
    if n == 0:
        return "Band 0 (0)"
    if n <= 4:
        return "Band 1 (1-4)"
    if n <= 19:
        return "Band 2 (5-19)"
    if n <= 49:
        return "Band 3 (20-49)"
    return "Band 4 (50+)"


# --------------------------------------------------------------------------
# Statistical helpers
# --------------------------------------------------------------------------

def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation.

    Effect size for the Wilcoxon signed-rank test, bounded in [-1, 1] and
    interpretable as the net proportion of pairs favouring `a`.
    """
    d = a - b
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    pos = r[d > 0].sum()
    neg = r[d < 0].sum()
    return float((pos - neg) / r.sum())


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Unpaired effect size, used for cross-dataset comparisons."""
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    # Rank-based computation; avoids the O(n*m) pairwise loop.
    all_v = np.concatenate([a, b])
    r = stats.rankdata(all_v)
    r_a = r[:n_a].sum()
    u_a = r_a - n_a * (n_a + 1) / 2
    return float(2 * u_a / (n_a * n_b) - 1)


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * n / (rank + 1)
        running = min(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def jonckheere_terpstra(groups: list[np.ndarray], direction: str = "decreasing"):
    """Test for a monotone trend across ordered groups.

    Returns (statistic, z, p). Uses the normal approximation with a tie
    correction omitted, which is conservative for the sample sizes here.
    """
    groups = [np.asarray(g) for g in groups if len(g) > 0]
    if len(groups) < 3:
        return (np.nan, np.nan, np.nan)

    jt = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            u = stats.mannwhitneyu(groups[j], groups[i],
                                   alternative="two-sided").statistic
            jt += u

    ns = np.array([len(g) for g in groups], dtype=float)
    N = ns.sum()
    mean = (N ** 2 - (ns ** 2).sum()) / 4.0
    var = (N ** 2 * (2 * N + 3) - (ns ** 2 * (2 * ns + 3)).sum()) / 72.0
    if var <= 0:
        return (jt, np.nan, np.nan)
    z = (jt - mean) / np.sqrt(var)
    if direction == "decreasing":
        p = stats.norm.cdf(z)
    elif direction == "increasing":
        p = stats.norm.sf(z)
    else:
        p = 2 * stats.norm.sf(abs(z))
    return (float(jt), float(z), float(p))


def ols_hc1(y: np.ndarray, x: np.ndarray):
    """Simple regression with HC1 robust standard errors.

    Per-user metric differences are heteroskedastic by construction (variance
    shrinks as volume grows), so classical standard errors would be wrong.
    """
    X = np.column_stack([np.ones_like(x), x])
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    S = (X * (resid ** 2)[:, None]).T @ X
    cov = XtX_inv @ S @ XtX_inv * (n / max(n - k, 1))
    se = np.sqrt(np.diag(cov))
    t = beta / np.where(se > 0, se, np.nan)
    p = 2 * stats.t.sf(np.abs(t), df=max(n - k, 1))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return {"intercept": float(beta[0]), "slope": float(beta[1]),
            "se_slope": float(se[1]), "t_slope": float(t[1]),
            "p_slope": float(p[1]), "r2": float(r2), "n": int(n)}


def bootstrap_stat(fn, arrays: list[np.ndarray], n_boot: int, rng,
                   alpha: float = 0.05):
    """Bootstrap a statistic over independent samples."""
    if n_boot <= 0:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n_boot):
        resampled = [a[rng.integers(0, len(a), len(a))] if len(a) else a
                     for a in arrays]
        vals.append(fn(*resampled))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


# --------------------------------------------------------------------------
# Per-user metric extraction (one scoring pass, then cached)
# --------------------------------------------------------------------------

def compute_per_user(dataset: str, proc_dir: Path, results_dir: Path,
                     args, B, H, E) -> pd.DataFrame:
    log(f"scoring pass for {dataset}", 2)
    D = B.load_dataset(proc_dir, dataset)
    csr, users_tbl, items_tbl = D["csr"], D["users"], D["items"]
    n_users, n_items = csr.shape

    item_train_count = np.zeros(n_items, dtype=np.float64)
    item_train_count[items_tbl["idx"].values] = items_tbl["train_count"].values
    user_train_count = np.zeros(n_users, dtype=np.float64)
    user_train_count[users_tbl["idx"].values] = users_tbl["train_count"].values

    R = B.binary_train_matrix(csr, D["train"], positives_only=True)
    F = B.build_item_features(dataset, D["features"], n_items,
                              args.max_text_features)
    factors = 32 if args.quick else args.factors
    cf_ctor = {
        "puresvd": lambda: B.PureSVD(factors=factors),
        "itemknn": lambda: B.ItemKNN(k=args.knn_neighbours),
        "als": lambda: B.ALS(factors=factors, iters=args.als_iters),
    }[args.cf_model]

    comps = {}
    for name, ctor in (("cf", cf_ctor), ("cb", lambda: B.ContentBased(F)),
                       ("pop", lambda: B.Popularity())):
        m = ctor()
        m.fit(R, {})
        comps[name] = m

    rng = np.random.default_rng(RNG_SEED)
    per_band = 200 if args.quick else args.eval_users_per_band
    gt = B.build_ground_truth(D["test"])
    sel_users, band_weights, band_pops = B.stratified_users(
        users_tbl.assign(in_test=users_tbl["idx"].isin(list(gt.keys()))),
        gt, per_band, rng)
    eval_ids = np.concatenate([sel_users[b] for b in BANDS_ORDER if len(sel_users[b])])
    nu = user_train_count[eval_ids]
    eval_bands = np.array([band_of(int(c)) for c in nu], dtype=object)

    n_per = 100 if args.quick else args.candidates
    max_cand = n_per * 3 + 200
    pack = H.build_candidates(comps, R, eval_ids, gt, n_per, max_cand,
                              args.batch, args.normalise)
    rel = H.relevance_matrix(pack["cand"], pack["mask"], eval_ids, gt)
    ni = item_train_count[pack["cand"].clip(min=0)]

    roster = {
        "popularity": ("static", {"w_cf": 0.0, "w_cb": 0.0}),
        "cf_only": ("static", {"w_cf": 1.0, "w_cb": 0.0}),
        "content_only": ("static", {"w_cf": 0.0, "w_cb": 1.0}),
    }
    bandwise_params = None
    calib_path = results_dir / dataset / "hybrid_calibration.json"
    if calib_path.exists():
        with open(calib_path) as fh:
            cal = json.load(fh)
        for fam, entry in cal.get("calibrated", {}).items():
            if fam == "bandwise":
                bandwise_params = entry["params"]
            else:
                roster[f"hybrid_{fam}"] = (fam, entry["params"])
    else:
        log(f"no calibration for {dataset}; hybrids omitted", 4)

    maxk = max(K_VALUES)
    rows = []
    roster_items = list(roster.items())
    if bandwise_params is not None:
        roster_items.append(("hybrid_bandwise", ("bandwise", bandwise_params)))

    for name, (family, params) in roster_items:
        if family == "bandwise":
            top = H.bandwise_apply(pack, nu, ni, eval_bands, params, maxk)
        else:
            top = H.score_and_rank(pack, nu, ni, family, params, maxk)

        pu = {}
        for k in K_VALUES:
            pu.update(E.per_user_metrics(top, rel, pack["n_pos"], k))

        frame = pd.DataFrame({
            "dataset": dataset, "model": name,
            "user_idx": eval_ids, "band": eval_bands,
            "train_count": nu, "n_pos": pack["n_pos"],
        })
        for metric, vals in pu.items():
            frame[metric] = vals
        rows.append(frame)
        log(f"{name}: ndcg@10={pu['ndcg@10'].mean():.4f}", 4)

    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------

def descriptives(df: pd.DataFrame, metric: str, n_boot: int, rng) -> pd.DataFrame:
    rows = []
    for (ds, band, model), g in df.groupby(["dataset", "band", "model"], sort=False):
        v = g[metric].values
        lo, hi = bootstrap_stat(lambda x: x.mean(), [v], n_boot, rng)
        rows.append({"dataset": ds, "band": band, "model": model,
                     "metric": metric, "mean": float(v.mean()),
                     "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                     "median": float(np.median(v)),
                     "ci_low": lo, "ci_high": hi, "n": int(len(v))})
    out = pd.DataFrame(rows)
    out["band"] = pd.Categorical(out["band"], BANDS_ORDER, ordered=True)
    return out.sort_values(["dataset", "band", "model"])


def omnibus(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for (ds, band), g in df.groupby(["dataset", "band"], sort=False):
        wide = g.pivot_table(index="user_idx", columns="model", values=metric)
        wide = wide.dropna()
        if wide.shape[1] < 3 or wide.shape[0] < 10:
            continue
        stat, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
        # Kendall's W as the effect size for a Friedman design.
        n, k = wide.shape
        w = stat / (n * (k - 1)) if n * (k - 1) > 0 else np.nan
        rows.append({"dataset": ds, "band": band, "metric": metric,
                     "friedman_chi2": float(stat), "p_value": float(p),
                     "kendalls_w": float(w), "n_users": int(n),
                     "n_models": int(k)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["band"] = pd.Categorical(out["band"], BANDS_ORDER, ordered=True)
        out = out.sort_values(["dataset", "band"])
    return out


def pairwise(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for (ds, band), g in df.groupby(["dataset", "band"], sort=False):
        wide = g.pivot_table(index="user_idx", columns="model", values=metric).dropna()
        models = list(wide.columns)
        cell = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a = wide[models[i]].values
                b = wide[models[j]].values
                d = a - b
                if np.allclose(d, 0):
                    p = 1.0
                else:
                    try:
                        p = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
                    except ValueError:
                        p = float("nan")
                cell.append({
                    "dataset": ds, "band": band, "metric": metric,
                    "model_a": models[i], "model_b": models[j],
                    "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                    "mean_diff": float(d.mean()),
                    "rank_biserial": rank_biserial(a, b),
                    "p_raw": p, "n_users": int(len(a)),
                })
        if cell:
            ps = np.array([c["p_raw"] for c in cell])
            hp = holm(np.nan_to_num(ps, nan=1.0))
            for c, v in zip(cell, hp):
                c["p_holm_within_cell"] = float(v)
            rows.extend(cell)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_bh_global"] = benjamini_hochberg(
            np.nan_to_num(out["p_raw"].values, nan=1.0))
        out["band"] = pd.Categorical(out["band"], BANDS_ORDER, ordered=True)
        out = out.sort_values(["dataset", "band", "model_a", "model_b"])
    return out


def volume_trend(df: pd.DataFrame, metric: str, treatment: str,
                 control: str) -> pd.DataFrame:
    """Does the treatment's advantage decline as interaction volume rises?"""
    rows = []
    for ds, g in df.groupby("dataset", sort=False):
        wide = g.pivot_table(index=["user_idx", "band", "train_count"],
                             columns="model", values=metric).reset_index()
        if treatment not in wide.columns or control not in wide.columns:
            continue
        wide = wide.dropna(subset=[treatment, control])
        wide["advantage"] = wide[treatment] - wide[control]

        groups = [wide.loc[wide["band"] == b, "advantage"].values
                  for b in BANDS_ORDER]
        jt, z, p_jt = jonckheere_terpstra(groups, direction="decreasing")

        x = np.log1p(wide["train_count"].values.astype(float))
        reg = ols_hc1(wide["advantage"].values.astype(float), x)

        band_means = {b: float(np.mean(v)) if len(v) else np.nan
                      for b, v in zip(BANDS_ORDER, groups)}

        rows.append({
            "dataset": ds, "metric": metric,
            "treatment": treatment, "control": control,
            "jt_statistic": jt, "jt_z": z, "jt_p_decreasing": p_jt,
            "ols_slope_log1p_volume": reg["slope"],
            "ols_se_hc1": reg["se_slope"],
            "ols_t": reg["t_slope"], "ols_p": reg["p_slope"],
            "ols_r2": reg["r2"], "n_users": reg["n"],
            **{f"advantage_{b}": v for b, v in band_means.items()},
        })
    return pd.DataFrame(rows)


def diff_in_diff(df: pd.DataFrame, metric: str, treatment: str,
                 control: str, n_boot: int, rng) -> pd.DataFrame:
    """Cold-minus-warm advantage, per dataset and differenced across them."""
    per_ds = {}
    rows = []
    for ds, g in df.groupby("dataset", sort=False):
        wide = g.pivot_table(index=["user_idx", "band"], columns="model",
                             values=metric).reset_index()
        if treatment not in wide.columns or control not in wide.columns:
            continue
        wide = wide.dropna(subset=[treatment, control])
        wide["adv"] = wide[treatment] - wide[control]
        cold = wide.loc[wide["band"].isin(COLD_BANDS), "adv"].values
        warm = wide.loc[~wide["band"].isin(COLD_BANDS), "adv"].values
        if len(cold) == 0 or len(warm) == 0:
            continue
        did = float(cold.mean() - warm.mean())
        lo, hi = bootstrap_stat(lambda c, w: c.mean() - w.mean(),
                                [cold, warm], n_boot, rng)
        u = stats.mannwhitneyu(cold, warm, alternative="two-sided")
        rows.append({
            "dataset": ds, "metric": metric,
            "advantage_cold": float(cold.mean()),
            "advantage_warm": float(warm.mean()),
            "diff_in_diff": did, "ci_low": lo, "ci_high": hi,
            "cliffs_delta": cliffs_delta(cold, warm),
            "mannwhitney_p": float(u.pvalue),
            "n_cold": int(len(cold)), "n_warm": int(len(warm)),
        })
        per_ds[ds] = (cold, warm)

    if len(per_ds) == 2:
        (d1, (c1, w1)), (d2, (c2, w2)) = per_ds.items()
        did1 = c1.mean() - w1.mean()
        did2 = c2.mean() - w2.mean()
        lo, hi = bootstrap_stat(
            lambda a, b, c, d: (a.mean() - b.mean()) - (c.mean() - d.mean()),
            [c1, w1, c2, w2], n_boot, rng)
        rows.append({
            "dataset": f"{d1} minus {d2}", "metric": metric,
            "advantage_cold": np.nan, "advantage_warm": np.nan,
            "diff_in_diff": float(did1 - did2), "ci_low": lo, "ci_high": hi,
            "cliffs_delta": np.nan, "mannwhitney_p": np.nan,
            "n_cold": int(len(c1) + len(c2)), "n_warm": int(len(w1) + len(w2)),
        })
    return pd.DataFrame(rows)


def cross_dataset(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Do the datasets agree on how to rank the models, band by band?"""
    rows = []
    means = (df.groupby(["dataset", "band", "model"])[metric]
               .mean().reset_index())
    datasets = sorted(means["dataset"].unique())
    if len(datasets) < 2:
        return pd.DataFrame()
    for band in BANDS_ORDER:
        sub = means[means["band"] == band]
        piv = sub.pivot_table(index="model", columns="dataset", values=metric).dropna()
        if piv.shape[0] < 3:
            continue
        a = piv[datasets[0]].values
        b = piv[datasets[1]].values
        tau = stats.kendalltau(a, b)
        rho = stats.spearmanr(a, b)
        rows.append({
            "band": band, "metric": metric,
            "dataset_a": datasets[0], "dataset_b": datasets[1],
            "kendall_tau": float(tau.statistic), "kendall_p": float(tau.pvalue),
            "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue),
            "best_model_a": piv[datasets[0]].idxmax(),
            "best_model_b": piv[datasets[1]].idxmax(),
            "n_models": int(piv.shape[0]),
        })
    return pd.DataFrame(rows)


def tradeoff(results_dir: Path, datasets: list[str], desc: pd.DataFrame,
             metric: str) -> pd.DataFrame:
    """Accuracy gain against cold-item coverage, model by model."""
    rows = []
    for ds in datasets:
        cov_path = results_dir / ds / "results_coverage.csv"
        if not cov_path.exists():
            continue
        cov = pd.read_csv(cov_path)
        acc = (desc[(desc["dataset"] == ds) & (desc["band"].isin(COLD_BANDS))]
               .groupby("model")["mean"].mean())
        for _, r in cov.iterrows():
            if r["model"] not in acc.index:
                continue
            rows.append({
                "dataset": ds, "model": r["model"],
                f"cold_band_{metric}": float(acc[r["model"]]),
                "cold_item_coverage@10": float(r.get("cold_item_coverage@10", np.nan)),
                "catalogue_coverage@10": float(r.get("catalogue_coverage@10", np.nan)),
                "gini@10": float(r.get("gini@10", np.nan)),
            })
    out = pd.DataFrame(rows)
    if len(out) >= 3:
        corr_rows = []
        for ds, g in out.groupby("dataset"):
            if len(g) < 3:
                continue
            r = stats.spearmanr(g[f"cold_band_{metric}"], g["cold_item_coverage@10"])
            corr_rows.append({"dataset": ds, "model": "_SPEARMAN_",
                              f"cold_band_{metric}": float(r.statistic),
                              "cold_item_coverage@10": float(r.pvalue),
                              "catalogue_coverage@10": np.nan, "gini@10": np.nan})
        if corr_rows:
            out = pd.concat([out, pd.DataFrame(corr_rows)], ignore_index=True)
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def make_figures(df, desc, trend, metric, treatment, control, fig_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed — figures skipped", 2)
        return []

    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    datasets = sorted(df["dataset"].unique())

    # 1. Metric by band, per dataset, with intervals.
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = desc[desc["dataset"] == ds]
        for model, g in sub.groupby("model"):
            g = g.sort_values("band")
            xs = np.arange(len(g))
            ax.plot(xs, g["mean"], marker="o", label=model)
            if g["ci_low"].notna().all():
                ax.fill_between(xs, g["ci_low"], g["ci_high"], alpha=0.15)
        ax.set_xticks(np.arange(len(BANDS_ORDER)))
        ax.set_xticklabels([b.split(" ")[1] for b in BANDS_ORDER])
        ax.set_xlabel("interaction-volume band")
        ax.set_ylabel(metric)
        ax.set_title(ds)
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"{metric} by interaction-volume band")
    fig.tight_layout()
    p = fig_dir / "metric_by_band.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # 2. Advantage against volume.
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        g = df[df["dataset"] == ds]
        wide = g.pivot_table(index=["user_idx", "train_count"],
                             columns="model", values=metric).reset_index()
        if treatment not in wide.columns or control not in wide.columns:
            continue
        wide = wide.dropna(subset=[treatment, control])
        wide["advantage"] = wide[treatment] - wide[control]
        bins = [-0.5, 0.5, 4.5, 19.5, 49.5, np.inf]
        wide["bin"] = pd.cut(wide["train_count"], bins, labels=BANDS_ORDER)
        m = wide.groupby("bin", observed=False)["advantage"].agg(["mean", "sem"])
        xs = np.arange(len(m))
        ax.bar(xs, m["mean"], yerr=1.96 * m["sem"].fillna(0), capsize=4)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([b.split(" ")[1] for b in BANDS_ORDER])
        ax.set_xlabel("interaction-volume band")
        ax.set_ylabel(f"{treatment} minus {control}")
        row = trend[trend["dataset"] == ds]
        if len(row):
            r = row.iloc[0]
            ax.set_title(f"{ds}\nJT p={r['jt_p_decreasing']:.3g}, "
                         f"OLS slope={r['ols_slope_log1p_volume']:+.4f}")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Hybrid advantage as a function of interaction volume")
    fig.tight_layout()
    p = fig_dir / "advantage_by_volume.png"
    fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    # 3. Cross-dataset consistency.
    if len(datasets) == 2:
        means = df.groupby(["dataset", "band", "model"])[metric].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 6))
        for band in BANDS_ORDER:
            sub = means[means["band"] == band]
            piv = sub.pivot_table(index="model", columns="dataset", values=metric).dropna()
            if piv.empty:
                continue
            ax.scatter(piv[datasets[0]], piv[datasets[1]],
                       label=band.split(" ")[1], s=40)
        lim = [0, max(means[metric].max() * 1.05, 1e-6)]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.5)
        ax.set_xlabel(f"{metric} — {datasets[0]}")
        ax.set_ylabel(f"{metric} — {datasets[1]}")
        ax.set_title("Cross-dataset agreement by band")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout()
        p = fig_dir / "cross_dataset.png"
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    log(f"{len(made)} figures written to {fig_dir}", 2)
    return made


# --------------------------------------------------------------------------
# Narrative summary
# --------------------------------------------------------------------------

def write_summary(path: Path, metric, treatment, control, desc, omni,
                  pair, trend, did, cross, tr):
    L = []
    A = L.append
    A(f"# Comparative statistical analysis\n")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M')}. "
      f"Primary metric: `{metric}`. Treatment: `{treatment}`. "
      f"Control: `{control}`.\n")

    A("## 1. Omnibus tests\n")
    if omni.empty:
        A("No cell had enough models for a Friedman test.\n")
    else:
        sig = omni[omni["p_value"] < 0.05]
        A(f"{len(sig)} of {len(omni)} dataset-band cells show a significant "
          f"difference among models (Friedman, p < 0.05). "
          f"Kendall's W ranges {omni['kendalls_w'].min():.3f}-"
          f"{omni['kendalls_w'].max():.3f}.\n")
        for _, r in omni.iterrows():
            A(f"- **{r['dataset']} / {r['band']}**: chi2={r['friedman_chi2']:.1f}, "
              f"p={r['p_value']:.3g}, W={r['kendalls_w']:.3f}, n={r['n_users']}")
        A("")

    A("## 2. Directional hypothesis: advantage declines with volume\n")
    if trend.empty:
        A("Treatment or control absent; trend not tested.\n")
    else:
        for _, r in trend.iterrows():
            verdict = ("supported" if (r["jt_p_decreasing"] < 0.05
                                       and r["ols_slope_log1p_volume"] < 0)
                       else "not supported")
            A(f"**{r['dataset']}** — {verdict}. "
              f"Jonckheere-Terpstra z={r['jt_z']:.2f}, "
              f"p(decreasing)={r['jt_p_decreasing']:.3g}. "
              f"Robust OLS on log1p(volume): slope="
              f"{r['ols_slope_log1p_volume']:+.4f} "
              f"(HC1 SE {r['ols_se_hc1']:.4f}, p={r['ols_p']:.3g}, "
              f"R2={r['ols_r2']:.3f}).")
            A(f"  Band means: " + ", ".join(
                f"{b.split(' ')[1]}={r.get(f'advantage_{b}', float('nan')):+.4f}"
                for b in BANDS_ORDER))
        A("")

    A("## 3. Cold versus warm, and across datasets\n")
    if did.empty:
        A("Not computed.\n")
    else:
        for _, r in did.iterrows():
            ci = (f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                  if pd.notna(r["ci_low"]) else "n/a")
            A(f"- **{r['dataset']}**: cold-warm difference "
              f"{r['diff_in_diff']:+.4f}, 95% CI {ci}"
              + (f", Cliff's delta={r['cliffs_delta']:+.3f}"
                 if pd.notna(r["cliffs_delta"]) else ""))
        A("\nAn interval excluding zero for the cross-dataset row indicates the "
          "cold-start advantage differs in size between datasets.\n")

    A("## 4. Cross-dataset consistency of model ordering\n")
    if cross.empty:
        A("Requires two datasets.\n")
    else:
        for _, r in cross.iterrows():
            A(f"- **{r['band']}**: Kendall tau={r['kendall_tau']:+.3f} "
              f"(p={r['kendall_p']:.3g}), best on {r['dataset_a']} = "
              f"`{r['best_model_a']}`, best on {r['dataset_b']} = "
              f"`{r['best_model_b']}`")
        A("")

    A("## 5. Strongest pairwise contrasts\n")
    if pair.empty:
        A("None computed.\n")
    else:
        top = pair.reindex(pair["rank_biserial"].abs().sort_values(
            ascending=False).index).head(12)
        A("| dataset | band | A | B | mean diff | rank-biserial | p (Holm) | p (BH) |")
        A("|---|---|---|---|---|---|---|---|")
        for _, r in top.iterrows():
            A(f"| {r['dataset']} | {r['band']} | {r['model_a']} | {r['model_b']} "
              f"| {r['mean_diff']:+.4f} | {r['rank_biserial']:+.3f} "
              f"| {r['p_holm_within_cell']:.3g} | {r['p_bh_global']:.3g} |")
        A("")

    A("## 6. Accuracy against cold-item coverage\n")
    if tr.empty:
        A("Coverage results not found; run 06_evaluate.py first.\n")
    else:
        sp = tr[tr["model"] == "_SPEARMAN_"]
        for _, r in sp.iterrows():
            A(f"- **{r['dataset']}**: Spearman rho between cold-band accuracy "
              f"and cold-item coverage = {r[f'cold_band_{metric}']:+.3f} "
              f"(p={r['cold_item_coverage@10']:.3g})")
        A("\nA strong negative correlation would indicate accuracy is being "
          "bought by routing away from cold items rather than by ranking them "
          "better.\n")

    A("## Caveats\n")
    A("- Users are sampled stratified by band; per-band figures are unbiased "
      "within band, and overall figures reweight by population share.")
    A("- Wilcoxon and Friedman treat per-user metrics as paired across models, "
      "which holds because every model scores the same users.")
    A("- p-values are reported raw, Holm-corrected within cell, and "
      "BH-corrected globally. State which is relied on.")
    A("- With thousands of users per band, small differences reach "
      "significance; effect sizes and intervals carry the argument.")

    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=Path("./data/processed"))
    ap.add_argument("--results", type=Path, default=None,
                    help="default: <processed>/../results")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <results>/analysis")
    ap.add_argument("--baselines", type=Path, default=Path("./04_baselines.py"))
    ap.add_argument("--hybrid", type=Path, default=Path("./05_hybrid.py"))
    ap.add_argument("--evaluate", type=Path, default=Path("./06_evaluate.py"))
    ap.add_argument("--datasets", nargs="+", default=["movielens", "amazon"])
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--treatment", default="hybrid_shrinkage")
    ap.add_argument("--control", default="cf_only")
    ap.add_argument("--from-cache", action="store_true",
                    help="reuse per_user_metrics.parquet, skip the scoring pass")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--cf-model", choices=["puresvd", "itemknn", "als"],
                    default="puresvd")
    ap.add_argument("--normalise", choices=["zscore", "minmax", "rank"],
                    default="zscore")
    ap.add_argument("--candidates", type=int, default=300)
    ap.add_argument("--eval-users-per-band", type=int, default=2000)
    ap.add_argument("--factors", type=int, default=128)
    ap.add_argument("--als-iters", type=int, default=15)
    ap.add_argument("--knn-neighbours", type=int, default=200)
    ap.add_argument("--max-text-features", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    results_dir = args.results or (args.processed.parent / "results")
    out_dir = args.out or (results_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "per_user_metrics.parquet"

    if args.from_cache or cache.exists():
        if not cache.exists():
            sys.exit(f"--from-cache given but {cache} does not exist")
        log(f"loading cached per-user metrics from {cache}")
        df = pd.read_parquet(cache)
    else:
        B = load_module(args.baselines, "baselines")
        H = load_module(args.hybrid, "hybrid")
        E = load_module(args.evaluate, "evaluate")
        frames = []
        for ds in args.datasets:
            if not (args.processed / ds).exists():
                log(f"{ds} not found under {args.processed}, skipping")
                continue
            frames.append(compute_per_user(ds, args.processed, results_dir,
                                           args, B, H, E))
        if not frames:
            sys.exit("no datasets processed")
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(cache, index=False)
        log(f"per-user metrics cached to {cache}")

    metric = args.metric
    if metric not in df.columns:
        sys.exit(f"metric '{metric}' not in cache. Available: "
                 f"{[c for c in df.columns if '@' in c]}")

    rng = np.random.default_rng(RNG_SEED)
    log("computing descriptives")
    desc = descriptives(df, metric, args.bootstrap, rng)
    desc.to_csv(out_dir / "descriptives.csv", index=False)

    log("omnibus tests")
    omni = omnibus(df, metric)
    omni.to_csv(out_dir / "omnibus_friedman.csv", index=False)

    log("pairwise post-hoc")
    pair = pairwise(df, metric)
    pair.to_csv(out_dir / "pairwise_posthoc.csv", index=False)

    log("volume trend")
    trend = volume_trend(df, metric, args.treatment, args.control)
    trend.to_csv(out_dir / "volume_trend.csv", index=False)

    log("difference in differences")
    did = diff_in_diff(df, metric, args.treatment, args.control,
                       args.bootstrap, rng)
    did.to_csv(out_dir / "diff_in_diff.csv", index=False)

    log("cross-dataset concordance")
    cross = cross_dataset(df, metric)
    cross.to_csv(out_dir / "cross_dataset.csv", index=False)

    log("accuracy-coverage trade-off")
    tr = tradeoff(results_dir, sorted(df["dataset"].unique()), desc, metric)
    tr.to_csv(out_dir / "tradeoff.csv", index=False)

    make_figures(df, desc, trend, metric, args.treatment, args.control,
                 out_dir / "figures")

    write_summary(out_dir / "analysis_summary.md", metric, args.treatment,
                  args.control, desc, omni, pair, trend, did, cross, tr)

    # ------------------------------------------------------------------
    print()
    print(f"=== {metric} by dataset and band ===")
    piv = desc.pivot_table(index=["dataset", "model"], columns="band",
                           values="mean", observed=False)
    print(piv.round(4).to_string())

    if not trend.empty:
        print(f"\n=== Directional test: {args.treatment} minus {args.control} ===")
        for _, r in trend.iterrows():
            verdict = "SUPPORTED" if (r["jt_p_decreasing"] < 0.05
                                      and r["ols_slope_log1p_volume"] < 0) else "not supported"
            print(f"{r['dataset']:<12} JT z={r['jt_z']:+.2f} "
                  f"p={r['jt_p_decreasing']:.3g} | "
                  f"OLS slope={r['ols_slope_log1p_volume']:+.5f} "
                  f"(p={r['ols_p']:.3g}) -> {verdict}")

    if not did.empty:
        print("\n=== Cold minus warm advantage ===")
        for _, r in did.iterrows():
            ci = (f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                  if pd.notna(r["ci_low"]) else "n/a")
            print(f"{r['dataset']:<24} {r['diff_in_diff']:+.4f}  95% CI {ci}")

    if not cross.empty:
        print("\n=== Cross-dataset rank concordance ===")
        for _, r in cross.iterrows():
            print(f"{r['band']:<16} tau={r['kendall_tau']:+.3f} "
                  f"(p={r['kendall_p']:.3g})  best: "
                  f"{r['best_model_a']} / {r['best_model_b']}")

    print(f"\nWritten: {out_dir}")
    print(f"Narrative summary: {out_dir / 'analysis_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
