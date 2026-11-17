"""Why did TSFM features give no detectable gain in the v0.1 sealed tests?

Descriptive diagnostics, not a search for a win and not a pre-registered test.
Nothing here trains a contestant or scores the test period. Inputs, all read
only:

- the four key runs' `results.json` and saved case files (the sealed test case
  files are used only for what they already contain: scores, case labels,
  history length, and the newly bad flag);
- the prepared transactions and their checklist feature cache, VALIDATION
  period only, for transaction-level measures;
- the TTM embedding caches, with PCA refitted on training days exactly as the
  harness does (`tells.features.tsfm_features.fit_pca`).

Every number in the output is computed here. Computed numbers are saved to a
JSON file so the Markdown can be re-rendered without recomputing.

Usage:
    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 uv run python scripts/diagnose_no_gain.py \
        [--numbers PATH.json] [--reuse] [--out docs/analysis/why-no-gain.md]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from cleanroom.metrics import budget_selection

from tells.features.checklist import FEATURE_COLUMNS
from tells.features.tsfm_features import _parts, fit_pca
from tells.harness import checklist_cache_path

RUNS = Path("artifacts/runs")
REF, NEW = "B_checklist_lgbm", "D_checklist_tsfm_ttm"
KEY_RUNS = {
    "amlworld_valid": "20261115T041547Z_amlworld_valid_f6ff26be_5e9b71",
    "amlworld_test": "20261116T190641Z_amlworld_test_f6ff26be_ac67fd",
    "tabformer_valid": "20261116T193405Z_tabformer_valid_701e5f22_d9a420",
    "tabformer_test": "20261117T045139Z_tabformer_test_701e5f22_2e74f8",
}
DATA = {"amlworld": "data/interim/amlworld.parquet", "tabformer": "data/interim/tabformer.parquet"}
SHUFFLE_DOC = Path("docs/analysis/tabformer-shuffle-scope-control.md")
SAMPLE_N = 40_000  # validation account-days per dataset for the redundancy regression
N_SHOW_PCS = 5
SEED = 0
Z95, Z80 = 1.959964, 0.841621  # two-sided 5% test; 80% power

# SHAP groups for diagnostic 4. "Short" = facts about the last 24 hours or about
# this transaction against the past (the kind of thing the midnight-cut TSFM
# summary cannot see for today); "long" = 7- to 30-day and lifetime summaries.
OWN = {
    "amount", "log_abs_amount", "is_refund", "hour", "weekday", "is_online", "is_chip",
    "has_error", "direction", "category_code", "channel_code",
}  # fmt: skip
SHORT = {
    "secs_since_prev", "n_tx_1h", "n_tx_24h", "sum_amount_24h", "is_new_merchant",
    "is_new_category", "n_new_counterparties_24h", "in_amount_24h", "out_amount_24h",
    "n_in_24h", "n_out_24h",
}  # fmt: skip
LONG = {
    "n_prior", "n_tx_7d", "n_tx_30d", "sum_amount_7d", "mean_amount_30d", "std_amount_30d",
    "max_amount_30d", "amount_vs_mean_30d", "amount_z_30d", "n_errors_7d", "share_online_30d",
    "n_new_counterparties_7d", "in_amount_7d", "out_amount_7d",
}  # fmt: skip


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def results(key: str) -> dict:
    return json.loads((RUNS / KEY_RUNS[key] / "results.json").read_text(encoding="utf-8"))


def cases(key: str, contestant: str) -> pl.DataFrame:
    return pl.read_parquet(RUNS / KEY_RUNS[key] / f"cases_{contestant}.parquet")


# ---------------------------------------------------------------- diagnostic 1
def d1_power() -> dict:
    out = {}
    for key in KEY_RUNS:
        s = results(key)
        r = s["comparisons"]["results"][NEW]
        n_new = s["contestants"][REF]["cases"]["n_first_positive_cases"]
        half = (r["ci_high"] - r["ci_low"]) / 2
        out[key] = {
            "budget": s["headline"]["budget"],
            "n_newly_bad": n_new,
            "b": r["reference_value"],
            "d": r["contestant_value"],
            "diff": r["diff"],
            "ci_low": r["ci_low"],
            "ci_high": r["ci_high"],
            "p_holm": r["p_holm"],
            "half_width": half,
            "mde80": half * (Z95 + Z80) / Z95,
            "half_width_accounts": half * n_new,
            "mde80_accounts": half * (Z95 + Z80) / Z95 * n_new,
            "diff_accounts": r["diff"] * n_new,
        }
    return out


# ---------------------------------------------------------------- diagnostic 2
def d2_reshuffle() -> dict:
    out = {}
    for key in ("amlworld_test", "tabformer_test", "amlworld_valid", "tabformer_valid"):
        s = results(key)
        budget = s["headline"]["budget"]
        b, d = cases(key, REF), cases(key, NEW)
        assert b.select("account_id", "day").equals(d.select("account_id", "day"))
        days = b["day"].to_numpy()
        first = b["is_first_positive"].to_numpy()
        sb = budget_selection(b["score"].to_numpy(), budget, days)
        sd = budget_selection(d["score"].to_numpy(), budget, days)
        cb, cd = (sb >= 0.5) & first, (sd >= 0.5) & first
        bs, ds = b["score"].to_numpy(), d["score"].to_numpy()
        rank_corr = (
            pl.DataFrame({"b": bs, "d": ds}).select(pl.corr("b", "d", method="spearman")).item()
        )
        # Overlap of the reviewed queues themselves (all cases, not only newly bad).
        qb, qd = sb >= 0.5, sd >= 0.5
        out[key] = {
            "budget": budget,
            "both": int((cb & cd).sum()),
            "only_b": int((cb & ~cd).sum()),
            "only_d": int((cd & ~cb).sum()),
            "neither": int((first & ~cb & ~cd).sum()),
            "pearson": float(np.corrcoef(bs, ds)[0, 1]),
            "spearman": float(rank_corr),
            "queue_overlap": float((qb & qd).sum() / max(qb.sum(), 1)),
        }
    return out


# ---------------------------------------------------------------- diagnostic 3
def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _r2(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """In-sample R squared and adjusted R squared of an OLS fit with intercept."""
    x = (x - x.mean(0)) / np.where(x.std(0) > 0, x.std(0), 1.0)
    a = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ coef
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    n, p = x.shape
    return float(r2), float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def _design(feats: pl.DataFrame) -> np.ndarray:
    """Checklist columns, signed-log transformed, nulls as 0 plus a missing flag, and squares."""
    cols = []
    for c in FEATURE_COLUMNS:
        v = feats[c].cast(pl.Float64).to_numpy()
        miss = np.isnan(v)
        v = _signed_log(np.where(miss, 0.0, v))
        cols += [v, v**2]
        if miss.any():
            cols.append(miss.astype(float))
    x = np.column_stack(cols)
    return x[:, x.std(0) > 0]


def d3_redundancy(dataset: str) -> dict:
    vkey, s = f"{dataset}_valid", results(f"{dataset}_valid")
    cache = Path(s["tsfm"]["ttm"]["cache"].replace("\\", "/"))
    train_end = date.fromisoformat(s["config"]["split"]["train"]["end"])
    k = s["tsfm"]["ttm"]["pca_components"]
    t0 = time.perf_counter()
    mean, scale, comps, explained = fit_pca(
        cache, lambda days: (days < train_end).to_numpy(), k, s["tsfm"]["ttm"]["pca_standardized"]
    )
    log(f"{dataset}: PCA refit in {time.perf_counter() - t0:.0f}s, explained {explained:.4f}")
    keys = cases(vkey, REF).select("account_id", "day").sample(n=SAMPLE_N, seed=SEED)
    rows = []
    for part in _parts(cache):
        pk = pl.read_parquet(part / "keys.parquet").with_row_index("row")
        hit = pk.join(keys, on=["account_id", "day"], how="inner")
        if hit.height:
            emb = np.load(part / "emb.npy", mmap_mode="r")
            idx = hit["row"].to_numpy()
            pcs = ((emb[idx].astype(np.float64) - mean) / scale) @ comps
            rows.append(
                hit.drop("row").with_columns(**{f"pc{i}": pl.Series(pcs[:, i]) for i in range(k)})
            )
    emb_tab = pl.concat(rows).filter(pl.col("has_history") == 1)
    log(f"{dataset}: {emb_tab.height} sampled validation account-days with history")
    sel = emb_tab.select("account_id", "day")
    day_min, day_max = sel["day"].min(), sel["day"].max()
    tx = (
        pl.scan_parquet(DATA[dataset])
        .select("tx_id", "account_id", "ts")
        .filter(pl.col("ts").dt.date().is_between(day_min, day_max))
        .with_columns(day=pl.col("ts").dt.date())
        .join(sel.lazy(), on=["account_id", "day"], how="inner")
        .collect()
    )
    feat_path = checklist_cache_path(s["provenance"]["data"]["transactions"]["sha256"])
    feats = (
        pl.scan_parquet(feat_path)
        .select("tx_id", *FEATURE_COLUMNS)
        .join(tx.lazy(), on="tx_id", how="inner")
        .collect()
    )
    casted = [pl.col(c).cast(pl.Float64) for c in FEATURE_COLUMNS]
    day_mean = feats.group_by("account_id", "day").agg([c.mean() for c in casted])
    day_first = (
        feats.sort("ts", "tx_id")
        .group_by("account_id", "day", maintain_order=True)
        .agg([c.first() for c in casted])
    )
    out = {
        "cache": str(cache),
        "pca_variance_explained": explained,
        "recorded_pca_variance_explained": s["tsfm"]["ttm"]["pca_variance_explained"],
        "n_sample": None,
        "per_pc": [],
    }
    # Each component's variance on the sample weights the average R squared.
    for label, table in (("mean", day_mean), ("first", day_first)):
        j = emb_tab.join(table, on=["account_id", "day"], how="inner").sort("account_id", "day")
        x = _design(j)
        hist = _signed_log(j["n_prior"].fill_nan(0).fill_null(0).to_numpy())
        xh = np.column_stack([hist, hist**2])
        out["n_sample"] = j.height
        r2s, adj, r2h, var = [], [], [], []
        for i in range(k):
            y = j[f"pc{i}"].to_numpy()
            r, a = _r2(y, x)
            r2s.append(r)
            adj.append(a)
            r2h.append(_r2(y, xh)[0])
            var.append(float(y.var()))
        w = np.array(var) / np.sum(var)
        out[label] = {
            "r2": r2s,
            "adj_r2": adj,
            "r2_history_only": r2h,
            "var": var,
            "weighted_adj_r2": float(np.dot(w, adj)),
            "weighted_r2_history_only": float(np.dot(w, r2h)),
            "n_predictors": int(x.shape[1]),
        }
    return out


# ---------------------------------------------------------------- diagnostic 4
def _shap_groups(expl: dict) -> dict:
    shap = expl["mean_abs_shap"]
    total = sum(shap.values())
    g = {"own": 0.0, "short": 0.0, "long": 0.0, "tsfm": 0.0, "other": 0.0}
    for name, v in shap.items():
        if name in OWN:
            g["own"] += v
        elif name in SHORT:
            g["short"] += v
        elif name in LONG:
            g["long"] += v
        elif name.startswith("ttm_"):
            g["tsfm"] += v
        else:
            g["other"] += v
    return {k: v / total for k, v in g.items()}


def _first_fraud_position(dataset: str) -> dict:
    """VALIDATION only: position of the first fraud transaction on each newly bad account's first bad day."""
    first = cases(f"{dataset}_valid", REF).filter(pl.col("is_first_positive"))
    sel = first.select("account_id", "day")
    tx = (
        pl.scan_parquet(DATA[dataset])
        .select("account_id", "ts", "label")
        .filter(pl.col("ts").dt.date().is_between(sel["day"].min(), sel["day"].max()))
        .with_columns(day=pl.col("ts").dt.date())
        .join(sel.lazy(), on=["account_id", "day"], how="inner")
        .collect()
    )
    per = tx.group_by("account_id", "day").agg(
        n_tx=pl.len(),
        first_fraud_ts=pl.col("ts").filter(pl.col("label") == 1).min(),
    )
    per = (
        per.join(tx, on=["account_id", "day"])
        .group_by("account_id", "day")
        .agg(
            pl.col("n_tx").first(),
            before=(pl.col("ts") < pl.col("first_fraud_ts")).sum(),
            same_ts=(pl.col("ts") == pl.col("first_fraud_ts")).sum(),
        )
    )
    before = per["before"].to_numpy()
    return {
        "n": per.height,
        "median_n_tx": float(per["n_tx"].median()),
        "median_before": float(np.median(before)),
        "mean_before": float(before.mean()),
        "share_first_of_day": float((before == 0).mean()),
        "share_le_2_before": float((before <= 2).mean()),
        "share_same_ts_group": float((per["same_ts"] > 1).mean()),
    }


def d4_same_day() -> dict:
    out = {}
    for key in KEY_RUNS:
        s = results(key)
        b = cases(key, REF)
        nb = b.filter(pl.col("is_first_positive"))
        hl = nb["history_len"].cast(pl.Float64)
        out[key] = {
            "n_newly_bad": nb.height,
            "share_hist_lt16": float((hl < 16).mean()),
            "median_hist": float(hl.median()),
            "all_cases_share_hist_lt16": float((b["history_len"] < 16).mean()),
            "median_n_tx_first_bad_day": float(nb["n_tx"].median()),
            "share_one_tx_first_bad_day": float((nb["n_tx"] == 1).mean()),
            "shap_b": _shap_groups(s["contestants"][REF]["explanation"]),
            "shap_d": _shap_groups(s["contestants"][NEW]["explanation"]),
            "tsfm_share_d": s["contestants"][NEW]["explanation"]["tsfm_share_of_attribution"],
        }
    for dataset in DATA:
        out[f"{dataset}_valid"]["first_fraud"] = _first_fraud_position(dataset)
        log(f"{dataset}: first-fraud positions done")
    return out


# ---------------------------------------------------------------- diagnostic 5
def d5_tsfm_alone() -> list:
    rows = []
    for f in sorted(RUNS.glob("*/results.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        cs = s.get("contestants", {})
        for name in cs:
            if not name.startswith("C_"):
                continue
            budget = s["headline"]["budget"]
            rows.append(
                {
                    "run_id": s["run_id"],
                    "part": s["part"],
                    "contestant": name,
                    "budget": budget,
                    "n_newly_bad": cs[name]["cases"]["n_first_positive_cases"],
                    "c": cs[name]["headline"],
                    "a": cs["A_lazy_amount"]["headline"],
                    "b": cs[REF]["headline"],
                    "random": budget,
                    "c_pr_auc": cs[name]["transactions"]["transaction_pr_auc"],
                    "b_pr_auc": cs[REF]["transactions"]["transaction_pr_auc"],
                    "base_rate": cs[name]["transactions"]["positive_rate"],
                    "pca_standardized": s.get("tsfm", {})
                    .get(name.removeprefix("C_tsfm_only_"), {})
                    .get("pca_standardized"),
                }
            )
    return rows


# ---------------------------------------------------------------- diagnostic 7
def d7_controls() -> dict:
    out = {"builtin": {}}
    for key in KEY_RUNS:
        c = results(key)["controls"]["label_permutation"]
        out["builtin"][key] = {
            k: c[k] for k in ("permuted_headline", "random_headline", "difference", "ci_low",
                              "ci_high", "consistent_with_chance")
        }  # fmt: skip
    rows = []
    pat = re.compile(
        r"^\| (\w+) \| ([\d.]+)% \| ([\d.]+)% \| ([\d.]+)% \| ([+-][\d.]+) \| \[([+-][\d.]+), ([+-][\d.]+)\] \|$"
    )
    for line in SHUFFLE_DOC.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            rows.append(
                {
                    "shuffle": m[1],
                    "budget": float(m[2]) / 100,
                    "model": float(m[3]) / 100,
                    "random": float(m[4]) / 100,
                    "diff": float(m[5]) / 100,
                    "ci_low": float(m[6]) / 100,
                    "ci_high": float(m[7]) / 100,
                }
            )
    out["shuffle_scope"] = rows
    return out


# ------------------------------------------------------------------- rendering
def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def num(x: float, d: int = 0) -> str:
    return f"{x:,.{d}f}"


DS_NAMES = {"amlworld": "AMLWorld", "tabformer": "TabFormer"}
NAMES = {
    "amlworld_valid": "AMLWorld validation",
    "amlworld_test": "AMLWorld sealed test",
    "tabformer_valid": "TabFormer validation",
    "tabformer_test": "TabFormer sealed test",
}


def _few(share: float) -> str:
    return "none" if share == 0 else f"only {100 * share:.1f}%"


def _amount(r2: float) -> str:
    if r2 >= 0.6:
        return "large"
    if r2 >= 0.3:
        return "moderate"
    return "modest"


def render(n: dict) -> str:
    p, r, red, sd, c5, c7 = n["d1"], n["d2"], n["d3"], n["d4"], n["d5"], n["d7"]
    at, tt, av, tv = (p[k] for k in ("amlworld_test", "tabformer_test", "amlworld_valid",
                                      "tabformer_valid"))  # fmt: skip
    L: list[str] = []
    a = L.append
    a("# Why the TSFM features gave no detectable gain (Tells v0.1)")
    a("")
    a(
        "Descriptive and explanatory, not pre-registered, and not a search for a win: no "
        "contestant was retrained and the test period was not scored again. Sources: the four key "
        "runs' `results.json` and case files (sealed test case files used only for what they "
        "already contain), the validation periods of the prepared data, and the TTM embedding "
        "caches. Every number is computed by `scripts/diagnose_no_gain.py`. Runs: "
        + ", ".join(f"{NAMES[k]} `{v}`" for k, v in KEY_RUNS.items())
        + "."
    )
    a("")
    a(
        "Terms used below. **B** is the bank-style model (checklist red flags into LightGBM). "
        "**D** is B plus TTM columns (16 PCA components of the TTM summary of each account's "
        "history before midnight, two surprise scores, a has-history flag). **C** is the TTM "
        "columns alone. A **newly bad account** is one whose first fraud day falls in the "
        "evaluated period; the headline is the share of them caught on that first day within the "
        "daily review budget."
    )
    a("")

    # 1
    a("## 1. Statistical power: how small a gain could each test see?")
    a("")
    a(
        "The 95% interval half-width is roughly the smallest difference that, if observed, would "
        "have been clearly separated from zero. A true gain needs to be about 1.4 times that "
        "(2.80 / 1.96 standard errors) to be detected 80% of the time. Converting points to "
        "accounts shows how few extra catches a gain would mean."
    )
    a("")
    a(
        "| Run | Budget | Newly bad | B | D | D minus B [95% interval] | Half-width | Gain needed for 80% power | That gain in accounts |"
    )
    a("|---|---|---|---|---|---|---|---|---|")
    for k in ("amlworld_test", "tabformer_test", "amlworld_valid", "tabformer_valid"):
        x = p[k]
        a(
            f"| {NAMES[k]} | {pct(x['budget'])} | {num(x['n_newly_bad'])} | {pct(x['b'])} | "
            f"{pct(x['d'])} | {pts(x['diff'])} [{pts(x['ci_low'])}, {pts(x['ci_high'])}] | "
            f"{100 * x['half_width']:.1f} pts | {100 * x['mde80']:.1f} pts | "
            f"about {num(x['mde80_accounts'])} of {num(x['n_newly_bad'])} |"
        )
    a("")
    a(
        f"**What it means.** AMLWorld could see a gain of about {100 * at['half_width']:.1f} points "
        f"({num(at['half_width_accounts'])} more of its {num(at['n_newly_bad'])} newly bad accounts); "
        f"the observed difference was {pts(at['diff'])} points, about {num(at['diff_accounts'])} "
        "accounts, so on AMLWorld the TSFM effect at the headline budget is genuinely close to "
        "zero, not hidden by noise. TabFormer had only "
        f"{num(tt['n_newly_bad'])} newly bad cards, so its half-width is "
        f"{100 * tt['half_width']:.1f} points and a real gain would need to be about "
        f"{100 * tt['mde80']:.1f} points (about {num(tt['mde80_accounts'])} extra cards) to be "
        f"detected reliably. Its observed {pts(tt['diff'])} points (about "
        f'{num(tt["diff_accounts"])} cards) is below that line: on TabFormer, "no detectable '
        'gain" is partly a power problem. Analogy: a bathroom scale that reads in 3-kilogram '
        "steps cannot tell you whether you gained one kilogram."
    )
    a("")

    # 2
    a("## 2. Reshuffle versus add: does D catch new accounts or different ones?")
    a("")
    a(
        "Same rule as `scripts/analyze_test_run.py`: a newly bad account is caught when at least "
        "half of its review slot is used (`cleanroom.metrics.budget_selection`, per day, ties share "
        "slots). Queue overlap is the share of B's reviewed cases that D also reviews."
    )
    a("")
    a(
        "| Run | Budget | Both | Only B | Only D | Neither | Score correlation (Pearson / Spearman) | Queue overlap |"
    )
    a("|---|---|---|---|---|---|---|---|")
    for k in ("amlworld_test", "tabformer_test", "amlworld_valid", "tabformer_valid"):
        x = r[k]
        a(
            f"| {NAMES[k]} | {pct(x['budget'])} | {num(x['both'])} | {num(x['only_b'])} | "
            f"{num(x['only_d'])} | {num(x['neither'])} | {x['pearson']:.3f} / {x['spearman']:.3f} | "
            f"{pct(x['queue_overlap'])} |"
        )
    ra, rt = r["amlworld_test"], r["tabformer_test"]
    a("")
    a(
        f"**What it means.** On the AMLWorld test, D gained {num(ra['only_d'])} accounts B missed "
        f"and lost {num(ra['only_b'])} that B caught, so the net change is close to zero while "
        f"{num(ra['only_b'] + ra['only_d'])} accounts changed hands. On TabFormer D gained "
        f"{num(rt['only_d'])} and lost {num(rt['only_b'])}: more gained than lost, but still a "
        "swap rather than a pure addition. The reviewed queues themselves differ a lot: only "
        f"{pct(ra['queue_overlap'], 0)} (AMLWorld) and {pct(rt['queue_overlap'], 0)} (TabFormer) "
        "of the cases B reviews are also reviewed by D, even though the two models' scores "
        "correlate strongly overall. Adding TTM columns does not stack new "
        "catches on top of B's; it swaps some catches for others, because the extra columns "
        "change which cases fill a fixed number of review slots. Analogy: a new scout on a "
        "team with a fixed roster size; every player the scout adds pushes someone else off "
        "the roster, and the team gets better only if the newcomers are better on average than "
        "the ones they replace."
    )
    a("")

    # 3
    a("## 3. Redundancy: how much of the TTM summary does the checklist already know?")
    a("")
    a(
        f"For a random sample of {num(SAMPLE_N)} validation account-days per dataset (seed {SEED}; "
        "only those with some history kept), each TTM PCA component was regressed on the "
        "account-day's checklist features. PCA was refitted on training days with the harness's "
        "own `fit_pca`; the share of variance it keeps matches the run's recorded value, which "
        "confirms these are the same components the contestants used. Checklist features are "
        "averaged over the day's transactions (main result) or taken from the day's first "
        "transaction (whose history is closest to the TSFM's midnight cut). Each of the 34 "
        "checklist columns enters as a signed log, its square, and a missing flag where needed; "
        "the fit is ordinary least squares, so R squared is a lower bound: a flexible model "
        "could explain at least this much. Category and channel are not included."
    )
    a("")
    for ds in ("amlworld", "tabformer"):
        x = red[ds]
        m, f = x["mean"], x["first"]
        a(
            f"**{DS_NAMES[ds]}** ({num(x['n_sample'])} account-days, {m['n_predictors']} "
            f"predictors; PCA keeps {pct(x['pca_variance_explained'])} of the variance, run "
            f"recorded {pct(x['recorded_pca_variance_explained'])}):"
        )
        a("")
        a(
            "| Component | Share of the 16 components' variance | Adjusted R squared (day mean) | Adjusted R squared (first transaction) | R squared from history length alone |"
        )
        a("|---|---|---|---|---|")
        tot = sum(m["var"])
        for i in range(N_SHOW_PCS):
            a(
                f"| PC{i} | {pct(m['var'][i] / tot)} | {m['adj_r2'][i]:.2f} | "
                f"{f['adj_r2'][i]:.2f} | {m['r2_history_only'][i]:.2f} |"
            )
        a(
            f"| All 16, variance-weighted | 100% | {m['weighted_adj_r2']:.2f} | "
            f"{f['weighted_adj_r2']:.2f} | {m['weighted_r2_history_only']:.2f} |"
        )
        a("")
    ra3, rt3 = red["amlworld"]["mean"], red["tabformer"]["mean"]
    top_a = max(range(len(ra3["adj_r2"])), key=lambda i: ra3["adj_r2"][i])
    big_a = max(range(len(ra3["var"])), key=lambda i: ra3["var"][i])
    a(
        "**What it means.** A high R squared means that component is mostly a re-description of "
        "facts the checklist already has (how many earlier events, typical amounts, gaps "
        f"between transactions). Weighted by variance, the checklist explains "
        f"{pct(ra3['weighted_adj_r2'], 0)} of the TTM summary on AMLWorld and "
        f"{pct(rt3['weighted_adj_r2'], 0)} on TabFormer with a linear fit, so the overlap is "
        f"{_amount(ra3['weighted_adj_r2'])} on AMLWorld and {_amount(rt3['weighted_adj_r2'])} on "
        f"TabFormer, not total. Some single components are close copies (AMLWorld PC{top_a}: "
        f"{ra3['adj_r2'][top_a]:.2f}), and history length alone explains "
        f"{pct(ra3['weighted_r2_history_only'], 0)} (AMLWorld) and "
        f"{pct(rt3['weighted_r2_history_only'], 0)} (TabFormer) of the summary. The rest is "
        "information the checklist does not hold in linear form (a flexible model might find "
        "more of it), but section 5 shows that the TTM summary as a whole carries little fraud "
        "signal, and section 4 shows that in D the TTM columns mainly took attribution away from "
        "the checklist's history features. On AMLWorld the component with the most variance on "
        f"these validation days is PC{big_a}, not PC0, although PCA orders components by their "
        "variance on training days: another sign of the training versus evaluation history "
        "shift (methodology M41). Analogy: a second witness who partly repeats the first and "
        "otherwise talks about things unrelated to the crime adds little to the case."
    )
    a("")

    # 4
    a("## 4. Same-day blindness: the TSFM cannot see the day it is scoring")
    a("")
    a(
        "The TTM summary is cut at midnight (methodology M32), so it knows nothing about the "
        "first bad day itself. Newly bad accounts and their histories, from the case files "
        "(`history_len` is the number of events before the account's first transaction that "
        "day):"
    )
    a("")
    a(
        "| Run | Newly bad | With fewer than 16 prior events | Median prior events | All cases with fewer than 16 | Median transactions on first bad day | First bad day with one transaction |"
    )
    a("|---|---|---|---|---|---|---|")
    for k in KEY_RUNS:
        x = sd[k]
        a(
            f"| {NAMES[k]} | {num(x['n_newly_bad'])} | {pct(x['share_hist_lt16'])} | "
            f"{num(x['median_hist'])} | {pct(x['all_cases_share_hist_lt16'])} | "
            f"{num(x['median_n_tx_first_bad_day'])} | {pct(x['share_one_tx_first_bad_day'])} |"
        )
    a("")
    a(
        "Where the first fraud transaction sits within the first bad day (validation periods "
        'only, transaction labels; "before" counts transactions with a strictly earlier '
        "timestamp, which is what the checklist can see):"
    )
    a("")
    a(
        "| Dataset (validation) | Newly bad | Median transactions that day | Median transactions before the first fraud one | First fraud is the day's first transaction | At most 2 before | First fraud shares its timestamp with another transaction |"
    )
    a("|---|---|---|---|---|---|---|")
    for ds in DATA:
        x = sd[f"{ds}_valid"]["first_fraud"]
        a(
            f"| {DS_NAMES[ds]} | {num(x['n'])} | {num(x['median_n_tx'])} | "
            f"{num(x['median_before'])} | {pct(x['share_first_of_day'])} | "
            f"{pct(x['share_le_2_before'])} | {pct(x['share_same_ts_group'])} |"
        )
    a("")
    a(
        "Where B's and D's attribution comes from (mean absolute SHAP shares on up to 20,000 "
        'evaluated transactions, from `results.json`). "This transaction" is the transaction\'s '
        'own facts (amount, hour, weekday, direction, category, channel, flags); "short window" '
        "is the last 24 hours or this transaction against the past (seconds since previous, "
        "counts and sums over 1 and 24 hours, new merchant or category, new counterparties in "
        '24 hours, money in and out over 24 hours); "long history" is 7- and 30-day and '
        "lifetime summaries."
    )
    a("")
    a(
        "| Run | B: this transaction | B: short window | B: long history | D: this transaction | D: short window | D: long history | D: TTM columns |"
    )
    a("|---|---|---|---|---|---|---|---|")
    for k in KEY_RUNS:
        b_, d_ = sd[k]["shap_b"], sd[k]["shap_d"]
        a(
            f"| {NAMES[k]} | {pct(b_['own'])} | {pct(b_['short'])} | {pct(b_['long'])} | "
            f"{pct(d_['own'])} | {pct(d_['short'])} | {pct(d_['long'])} | {pct(d_['tsfm'])} |"
        )
    st, sv = sd["amlworld_test"], sd["amlworld_valid"]["first_fraud"]
    tf = sd["tabformer_valid"]["first_fraud"]
    bt = st["shap_b"]
    a("")
    a(
        f"**What it means.** On AMLWorld, B draws {pct(bt['own'] + bt['short'], 0)} of its "
        "attribution from the transaction itself and the last 24 hours: the part a "
        "midnight-cut history summary cannot see (today) or sees only blurred into a long "
        f"average (the hours before midnight). In AMLWorld validation, {pct(sv['share_first_of_day'], 0)} "
        "of newly bad accounts' first fraud transaction is the first transaction of the day "
        f"(TabFormer: {pct(tf['share_first_of_day'], 0)}), so the decisive evidence is usually the "
        "fraud transaction's own facts, which B already reads. And "
        f"{pct(st['share_hist_lt16'], 0)} of AMLWorld test newly bad accounts had fewer than 16 "
        "earlier events (16 events is one Chronos-2 patch; TTM reads a 512-step window in "
        "64-step patches, so for these accounts its input is almost all padding, M35 and M45), "
        "so for many of them the summary has little to read. TabFormer is the opposite case: long histories "
        f"({_few(sd['tabformer_test']['share_hist_lt16'])} of the test's newly bad cards had fewer than 16), "
        "and that is where D's "
        "point estimate was a little better. Analogy: a background check run last night cannot "
        "flag what happens at the counter this morning; the cashier watching the counter can."
    )
    a("")
    dt = sd["amlworld_test"]["shap_d"]
    hist_b, hist_d = bt["short"] + bt["long"], dt["short"] + dt["long"]
    tbt, tdt = sd["tabformer_test"]["shap_b"], sd["tabformer_test"]["shap_d"]
    a(
        f"The attribution table also shows where D's TTM share ({pct(dt['tsfm'])} on the AMLWorld "
        f"test) came from: the history features' share fell from {pct(hist_b)} in B to "
        f"{pct(hist_d)} in D, while the transaction's own facts stayed at {pct(bt['own'])} and "
        f"{pct(dt['own'])}. On the TabFormer test the TTM share was {pct(tdt['tsfm'])} and the "
        f"history share fell from {pct(tbt['short'] + tbt['long'])} to "
        f"{pct(tdt['short'] + tdt['long'])}. The TTM columns took over work the checklist's "
        "history features were already doing, which fits the redundancy in section 3."
    )
    a("")

    # 5
    a("## 5. How much signal is in the TSFM alone?")
    a("")
    a(
        "Contestant C uses only the TSFM columns. Random ranking catches exactly the budget "
        "share. Every run with a C contestant is listed, including development runs on samples "
        "and the first, unstandardized TTM variant."
    )
    a("")
    a(
        "| Run | Part | Contestant | Budget | Newly bad | A (amount) | Random | C | B | C transaction PR-AUC | B transaction PR-AUC | Fraud base rate |"
    )
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in c5:
        a(
            f"| `{x['run_id']}` | {x['part']} | {x['contestant']} | {pct(x['budget'])} | "
            f"{num(x['n_newly_bad'])} | {pct(x['a'])} | {pct(x['random'])} | {pct(x['c'])} | "
            f"{pct(x['b'])} | {x['c_pr_auc']:.3f} | {x['b_pr_auc']:.3f} | {x['base_rate']:.4f} |"
        )
    key_c = {x["run_id"]: x for x in c5}
    cat, ctt = key_c[KEY_RUNS["amlworld_test"]], key_c[KEY_RUNS["tabformer_test"]]
    a("")
    vs_a = "below" if cat["c"] < cat["a"] else "above"
    a(
        f"**What it means.** On the sealed tests C caught {pct(cat['c'])} on AMLWorld (random "
        f"{pct(cat['random'])}, ranking by amount {pct(cat['a'])}, B {pct(cat['b'])}) and "
        f"{pct(ctt['c'])} on TabFormer (random {pct(ctt['random'])}, ranking by amount "
        f"{pct(ctt['a'])}, B {pct(ctt['b'])}). On the AMLWorld test the TSFM alone is {vs_a} "
        "simply ranking by amount. Its transaction PR-AUC is "
        f"{cat['c_pr_auc'] / cat['base_rate']:.1f} (AMLWorld) and "
        f"{ctt['c_pr_auc'] / ctt['base_rate']:.1f} (TabFormer) times the fraud base rate, "
        f"against {cat['b_pr_auc'] / cat['base_rate']:.0f} and "
        f"{ctt['b_pr_auc'] / ctt['base_rate']:.0f} times for B. So the summary carries some "
        "signal, but far less than the checklist, and a feature that weak can only help B if "
        "its signal is also different from what B already has. Analogy: a smoke detector that "
        "rarely beeps for real fires adds little to a building that already has sprinklers in "
        "every room."
    )
    a("")

    # 6
    a("## 6. Domain mismatch (cited hypothesis, not measured)")
    a("")
    a(
        "`docs/tsfm-notes.md` records Chronos-2's documented pretraining data as energy, traffic, "
        "weather, web, cloud traces, M4, and synthetic series, with no financial or transaction "
        "data listed; TTM r2's pretraining data was not checked in a paper. Those are mostly "
        "regularly sampled series with daily or weekly rhythms (our characterization, not a claim from the notes). Account histories are irregular: "
        "events arrive in bursts, with long silences between them, and the project feeds them as "
        "one step per event with the time gap as a separate channel (methodology M32, M35). The "
        "hypothesis is that a model pretrained on smooth, evenly spaced series summarizes such "
        "bursty event sequences poorly, so its summary falls back to coarse facts such as how "
        "long and how large the history is (consistent with section 3, but not proven by it). "
        "This was not measured here."
    )
    a("")

    # 7
    a("## 7. The label-permutation control: how easily is unusual fraud ranked without labels?")
    a("")
    a(
        "Built-in control (B retrained on labels shuffled within each training day), from each "
        "run's `results.json`, at the headline budget:"
    )
    a("")
    a("| Run | Noise-trained model | Random | Difference [95% interval] | Consistent with chance |")
    a("|---|---|---|---|---|")
    for k, x in c7["builtin"].items():
        a(
            f"| {NAMES[k]} | {pct(x['permuted_headline'])} | {pct(x['random_headline'])} | "
            f"{pts(x['difference'])} [{pts(x['ci_low'])}, {pts(x['ci_high'])}] | "
            f"{'yes' if x['consistent_with_chance'] else 'no'} |"
        )
    a("")
    a(
        "Shuffle-scope follow-up on TabFormer validation at the 0.1% headline budget (parsed from "
        f"`{SHUFFLE_DOC.as_posix()}`):"
    )
    a("")
    a("| Shuffle | Noise-trained model | Random | Difference [95% interval] |")
    a("|---|---|---|---|")
    head_rows = [x for x in c7["shuffle_scope"] if abs(x["budget"] - 0.001) < 1e-9]
    for x in head_rows:
        a(
            f"| {x['shuffle']} | {pct(x['model'])} | {pct(x['random'])} | {pts(x['diff'])} "
            f"[{pts(x['ci_low'])}, {pts(x['ci_high'])}] |"
        )
    across = [x for x in head_rows if x["shuffle"].startswith("across_all")]
    lo, hi = min(x["model"] for x in across), max(x["model"] for x in across)
    ctv = key_c[KEY_RUNS["tabformer_valid"]]
    within = next(x for x in head_rows if x["shuffle"] == "within_day")
    a("")
    a(
        "**What it means.** Even with labels shuffled across all training days, so no real link "
        f"between behavior and fraud is left, LightGBM catches {pct(lo)} to {pct(hi)} of newly "
        "bad TabFormer cards at 0.1%, against 0.1% for random ranking. One explanation consistent "
        "with this (not proven here): a tree fitted to noise still gives extreme scores to rare "
        "combinations of feature values, so it behaves like an anomaly detector, and fraud on "
        "TabFormer often looks unusual. For comparison, C (TTM alone, trained on real labels) "
        f"caught {pct(ctv['c'])} on the same period, between the across-all-days noise models "
        f"and the within-day one ({pct(within['model'])}): the TSFM's standalone signal is of "
        "the same order as what a tree finds with no real labels at all. On AMLWorld the "
        "noise-trained model was consistent with chance, so this reading applies to TabFormer "
        "only. Two consequences: unusual-looking fraud is already near the top of B's queue "
        "without any help, and a new feature can only add catches among the fraud that does not "
        "look unusual, which is a smaller and harder pool. Analogy: if a guard dog barks at "
        "every stranger, a second dog that also barks at strangers adds nothing; the help you "
        "need is with intruders who look like residents."
    )
    a("")

    # ranked reasons
    a("## Most likely reasons, ranked (interpretation of the numbers above)")
    a("")
    a(
        "The ranking is a judgment about how directly each number bears on the headline; the "
        "reasons overlap and none was tested as a cause."
    )
    a("")
    a(
        f"1. **The evidence that decides a first bad day is today's, which the TSFM cannot see.** "
        f"On the AMLWorld test B draws {pct(bt['own'] + bt['short'], 0)} of its attribution from "
        f"the transaction itself and the last 24 hours (TabFormer: "
        f"{pct(tbt['own'] + tbt['short'], 0)}), and in validation the first fraud transaction was "
        f"the day's first transaction for {pct(sv['share_first_of_day'], 0)} (AMLWorld) and "
        f"{pct(tf['share_first_of_day'], 0)} (TabFormer) of newly bad accounts (section 4)."
    )
    a(
        f"2. **The TTM summary alone carries little fraud signal.** C caught {pct(cat['c'])} on "
        f"the AMLWorld test (amount ranking {pct(cat['a'])}, B {pct(cat['b'])}) and "
        f"{pct(ctt['c'])} on the TabFormer test (B {pct(ctt['b'])}) (section 5)."
    )
    a(
        f"3. **In a fixed queue, D's new catches were cancelled by lost ones.** At the AMLWorld "
        f"headline D gained {num(ra['only_d'])} and lost {num(ra['only_b'])} newly bad accounts "
        f"(TabFormer: gained {num(rt['only_d'])}, lost {num(rt['only_b'])}), while only "
        f"{pct(ra['queue_overlap'], 0)} and {pct(rt['queue_overlap'], 0)} of B's reviewed cases "
        "stayed in D's queue (section 2). This is how the null shows up rather than a root "
        "cause, but it explains why a changed queue gave an unchanged headline."
    )
    a(
        f"4. **Part of the TTM summary repeats the checklist, and D used it as a substitute.** "
        f"The checklist explains {pct(ra3['weighted_adj_r2'], 0)} (AMLWorld) and "
        f"{pct(rt3['weighted_adj_r2'], 0)} (TabFormer) of the components' variance linearly "
        f"(section 3), and in D the history features' attribution fell from {pct(hist_b)} to "
        f"{pct(hist_d)} on the AMLWorld test as the TTM columns took {pct(dt['tsfm'])} "
        "(section 4)."
    )
    a(
        f"5. **On AMLWorld, many newly bad accounts have short histories.** "
        f"{pct(st['share_hist_lt16'], 0)} had fewer than 16 earlier events (median "
        f"{num(st['median_hist'])}), so a history summary has little to read (section 4); "
        "training-period accounts are shorter still (methodology M41)."
    )
    a(
        f"6. **On TabFormer, the test was too small to see a modest gain.** "
        f"{num(tt['n_newly_bad'])} newly bad cards give a half-width of "
        f"{100 * tt['half_width']:.1f} points; detecting a gain reliably needed about "
        f"{100 * tt['mde80']:.1f} points, and the observed difference was {pts(tt['diff'])} "
        f"(section 1). This does not apply to AMLWorld, where the half-width was "
        f"{100 * at['half_width']:.1f} points and the estimate {pts(at['diff'])}."
    )
    a(
        f"7. **Unusual-looking fraud is already easy to rank (TabFormer).** Models trained on "
        f"labels shuffled across all days still caught {pct(lo)} to {pct(hi)} at the 0.1% "
        f"budget, and TTM alone caught {pct(ctv['c'])} (section 7), which leaves less room for "
        "any new feature."
    )
    a(
        "8. **Domain mismatch between pretraining series and transaction histories** (section 6). "
        "Plausible and consistent with the above, but not measured."
    )
    a("")
    a("## What would have to change for a TSFM to help (proposals, not evidence)")
    a("")
    a(
        "These follow from the reasons above; none has been tested in this project.\n\n"
        "- **See the day being scored.** Embed the sequence up to each transaction (or each hour) "
        "instead of up to midnight, so the summary can react to a same-day burst. This costs one "
        "embedding per transaction instead of per account-day.\n"
        "- **Learn from fraud labels.** Fine-tune the TSFM, or train a small head on its patch "
        "states, so the summary is shaped toward fraud rather than toward forecasting energy "
        "or traffic.\n"
        "- **Keep the order and the recent past.** Average pooling over the whole history blurs "
        "a recent change; recent-patch pooling with more components is v0.2 idea 1 (M45).\n"
        "- **Larger models on full data,** which needs a GPU (Chronos-2 ran only on a sample, M36).\n"
        "- **Data with longer and more realistic histories,** ideally real transaction data under "
        "a data-sharing agreement, where an account's past is long enough to summarize and fraud "
        "is less obviously unusual.\n"
        "- **Enough newly bad accounts to see a few points of gain,** or a pre-registered "
        "tighter budget where section 1 of `docs/analysis/amlworld-test-descriptive.md` hinted "
        "at a difference."
    )
    a("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numbers", required=True, help="JSON file for computed numbers (scratch)")
    ap.add_argument("--reuse", action="store_true", help="render from saved numbers")
    ap.add_argument("--out", default="docs/analysis/why-no-gain.md")
    ap.add_argument("--only", default="", help="comma list of diagnostics to compute (debug)")
    args = ap.parse_args()
    numbers_path = Path(args.numbers)
    if args.reuse:
        n = json.loads(numbers_path.read_text(encoding="utf-8"))
    else:
        n = json.loads(numbers_path.read_text(encoding="utf-8")) if args.only else {}
        steps = {
            "d1": d1_power,
            "d2": d2_reshuffle,
            "d3": lambda: {ds: d3_redundancy(ds) for ds in DATA},
            "d4": d4_same_day,
            "d5": d5_tsfm_alone,
            "d7": d7_controls,
        }
        only = [s for s in args.only.split(",") if s] or list(steps)
        for name in only:
            t0 = time.perf_counter()
            n[name] = steps[name]()
            log(f"{name} done in {time.perf_counter() - t0:.0f}s")
            numbers_path.parent.mkdir(parents=True, exist_ok=True)
            numbers_path.write_text(json.dumps(n, indent=1, default=str), encoding="utf-8")
    out = Path(args.out)
    if all(k in n for k in ("d1", "d2", "d3", "d4", "d5", "d7")):
        out.write_text(render(n), encoding="utf-8")
        log(f"wrote {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
