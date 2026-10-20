"""Statistical analysis of decoding-strategy sensitivity. Reads results/scores/*.csv and writes
results/analysis/*.csv|json and figures/*.png.

Scores: factual/reasoning/code are 0/1 per item (shown as % points); creative is the mean
LLM-judge rating on a 1-10 scale.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, SCORE_DIR, STRATEGIES, TASKS  # noqa: E402

OUT = os.path.join(ROOT, "results", "analysis")
FIG = os.path.join(ROOT, "figures")
SAMPLED = [s for s in STRATEGIES if s != "greedy"]
N_BOOT = 1000
rng = np.random.default_rng(42)


def load():
    df = pd.read_csv(os.path.join(SCORE_DIR, "item_scores.csv"))
    df["id"] = df["id"].astype(str)
    jp = os.path.join(SCORE_DIR, "creative_judge.csv")
    if os.path.exists(jp):
        j = pd.read_csv(jp)[["model", "strategy", "seed", "id", "judge_mean"]]
        j["id"] = j["id"].astype(str)
        df = df.merge(j, on=["model", "strategy", "seed", "id"], how="left")
        cm = df.task == "creative"
        df.loc[cm, "score"] = df.loc[cm, "judge_mean"]
    df = df.dropna(subset=["score"])
    # Keep only (model, task) cells with the complete grid of 16 (strategy, seed) runs.
    cnt = df.groupby(["model", "task"])[["strategy", "seed"]].apply(lambda g: len(g.drop_duplicates()))
    full = set(cnt[cnt == 16].index)
    return df[[(m, t) in full for m, t in zip(df.model, df.task)]]


def scale(task):
    return 1.0 if task == "creative" else 100.0


def model_means(d):
    """model x strategy mean score (averaged over items, then over seeds)."""
    per_seed = d.groupby(["model", "strategy", "seed"])["score"].mean()
    return per_seed.groupby(["model", "strategy"]).mean().unstack()[list(STRATEGIES)]


def kendall_vs_greedy(mm):
    return {s: stats.kendalltau(mm["greedy"], mm[s]).statistic for s in SAMPLED}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    df = load()
    models = sorted(df.model.unique())
    print("models:", len(models), models)
    summary, per_model_rows, tau_rows, seed_rows, tests = {}, [], [], [], []

    for task in TASKS:
        d = df[df.task == task]
        if d.empty:
            continue
        k = scale(task)
        mm = model_means(d) * k
        mm.to_csv(os.path.join(OUT, f"model_means_{task}.csv"))
        delta = mm.sub(mm["greedy"], axis=0)
        rel = delta.div(mm["greedy"], axis=0) * 100
        for m in mm.index:
            for s in STRATEGIES:
                per_model_rows.append(dict(task=task, model=m, strategy=s, score=mm.loc[m, s],
                                           delta=delta.loc[m, s], rel_delta_pct=rel.loc[m, s]))
        # Across-model tests of delta vs greedy (Wilcoxon signed-rank; also t-test).
        for s in SAMPLED:
            w = stats.wilcoxon(delta[s]) if (delta[s] != 0).any() else None
            t = stats.ttest_1samp(delta[s], 0)
            tests.append(dict(task=task, strategy=s, mean_delta=delta[s].mean(), sd_delta=delta[s].std(),
                              min_delta=delta[s].min(), max_delta=delta[s].max(),
                              max_abs_delta=delta[s].abs().max(),
                              mean_rel_pct=rel[s].mean(), median_rel_pct=rel[s].median(),
                              t_p=t.pvalue, wilcoxon_p=(w.pvalue if w else 1.0)))

        # Kendall tau with item-bootstrap CI: resample items, recompute model means & rankings.
        piv = d.pivot_table(index="id", columns=["model", "strategy", "seed"], values="score")
        ids = piv.index.values
        base = kendall_vs_greedy(mm)
        boots = {s: [] for s in SAMPLED}
        mean_pair_boot = []
        for _ in range(N_BOOT):
            samp = piv.loc[rng.choice(ids, len(ids), replace=True)].mean()
            bm = samp.groupby(level=["model", "strategy"]).mean().unstack()[list(STRATEGIES)]
            for s, v in kendall_vs_greedy(bm).items():
                boots[s].append(v)
        # Seed-to-seed ranking stability within a sampled strategy (seed i vs seed j).
        ps = d.groupby(["model", "strategy", "seed"])["score"].mean().unstack()
        for s in STRATEGIES:
            lo, hi = (np.nanpercentile(boots[s], [2.5, 97.5]) if s in boots else (np.nan, np.nan))
            within = []
            if s != "greedy":
                x = ps.xs(s, level="strategy")
                within = [stats.kendalltau(x[a], x[b]).statistic for a, b in [(0, 1), (0, 2), (1, 2)]]
            tau_rows.append(dict(task=task, strategy=s, tau_vs_greedy=base.get(s, 1.0), ci_lo=lo, ci_hi=hi,
                                 frac_boot_gt_0_9=(np.mean(np.array(boots[s]) > 0.9) if s in boots else np.nan),
                                 tau_seed_vs_seed=np.mean(within) if within else np.nan))
        # Mean over all strategy pairs (alternative stability summary).
        pair_taus = [stats.kendalltau(mm[a], mm[b]).statistic
                     for i, a in enumerate(STRATEGIES) for b in list(STRATEGIES)[i + 1:]]
        # Seed variability: std over seeds of the run-level score, per model & sampled strategy.
        for s in SAMPLED:
            x = ps.xs(s, level="strategy") * k
            sub = d[d.strategy == s]
            # Per-item flip rate: fraction of items whose 0/1 outcome is not identical across seeds.
            itm = sub.pivot_table(index=["model", "id"], columns="seed", values="score")
            if task == "creative":
                flip = itm.std(axis=1).groupby(level="model").mean()  # item-level judge-score std
            else:
                flip = (itm.nunique(axis=1) > 1).groupby(level="model").mean() * 100
            for m in x.index:
                p = mm.loc[m, s] / k
                n = d.id.nunique()
                binom_se = np.sqrt(p * (1 - p) / n) * k if task != "creative" else np.nan
                seed_rows.append(dict(task=task, model=m, strategy=s, seed_std=x.loc[m].std(ddof=1),
                                      seed_range=x.loc[m].max() - x.loc[m].min(),
                                      seed_cv_pct=x.loc[m].std(ddof=1) / max(1e-9, x.loc[m].mean()) * 100,
                                      item_flip=flip.loc[m], binom_se=binom_se))
        summary[task] = dict(
            n_items=int(d.id.nunique()), n_models=len(mm),
            mean_by_strategy=mm.mean().round(3).to_dict(),
            mean_delta_vs_greedy=delta.mean().round(3).to_dict(),
            mean_rel_delta_pct=rel.mean().round(2).to_dict(),
            max_abs_delta_any_model_any_strategy=float(delta[SAMPLED].abs().max().max()),
            tau_vs_greedy={k_: round(v, 3) for k_, v in base.items()},
            mean_pairwise_tau_all_strategies=float(np.mean(pair_taus)),
            ranking_greedy=list(mm["greedy"].sort_values(ascending=False).index),
        )
        if task == "reasoning":
            gap = mm["greedy"] - mm["t1.0"]
            rec = (mm["topp0.9"] - mm["t1.0"]) / gap.where(gap.abs() > 1e-9)
            pooled_rec = (mm["topp0.9"].mean() - mm["t1.0"].mean()) / (mm["greedy"].mean() - mm["t1.0"].mean())
            summary[task]["greedy_minus_t1"] = gap.round(2).to_dict()
            summary[task]["topp_recovery_fraction_per_model"] = rec.round(3).to_dict()
            summary[task]["topp_recovery_fraction_pooled"] = float(pooled_rec)
            summary[task]["minp_recovery_fraction_pooled"] = float(
                (mm["minp0.1"].mean() - mm["t1.0"].mean()) / (mm["greedy"].mean() - mm["t1.0"].mean()))

    pm = pd.DataFrame(per_model_rows); pm.to_csv(os.path.join(OUT, "per_model_scores.csv"), index=False)
    tt = pd.DataFrame(tests)
    # Holm correction across all (task, strategy) Wilcoxon tests.
    order = np.argsort(tt.wilcoxon_p.values); m_ = len(order); adj = np.empty(m_); run = 0
    for r, i in enumerate(order):
        run = max(run, min(1, (m_ - r) * tt.wilcoxon_p.values[i])); adj[i] = run
    tt["wilcoxon_p_holm"] = adj
    tt.to_csv(os.path.join(OUT, "delta_tests.csv"), index=False)
    tau = pd.DataFrame(tau_rows); tau.to_csv(os.path.join(OUT, "kendall_tau.csv"), index=False)
    sd = pd.DataFrame(seed_rows); sd.to_csv(os.path.join(OUT, "seed_variability.csv"), index=False)

    # H5: compare seed variability across domains.
    seedsum = sd.groupby("task")[["seed_std", "seed_range", "seed_cv_pct", "item_flip", "binom_se"]].mean()
    seedsum["std_over_binomSE"] = (sd.seed_std / sd.binom_se).groupby(sd.task).mean()
    seedsum.to_csv(os.path.join(OUT, "seed_variability_by_task.csv"))
    acc_tasks = [t for t in ["factual", "reasoning", "code"] if t in sd.task.unique()]
    h5 = {}
    for metric in ["seed_std", "item_flip"]:
        # Per (model, strategy) cell, compare code vs each other accuracy task (paired Wilcoxon).
        piv = sd.pivot_table(index=["model", "strategy"], columns="task", values=metric)
        for t in acc_tasks:
            if t != "code" and "code" in piv:
                h5[f"{metric}: code vs {t}"] = dict(
                    code_mean=float(piv["code"].mean()), other_mean=float(piv[t].mean()),
                    wilcoxon_p=float(stats.wilcoxon(piv["code"], piv[t]).pvalue))
    summary["seed_variability_by_task"] = seedsum.round(3).to_dict()
    summary["h5_tests"] = h5
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=1, default=float)[:6000])
    print(tt.round(4).to_string())
    print(tau.round(3).to_string())
    print(seedsum.round(3).to_string())
    creative_extras(df)
    make_figures(pm, tau, sd)


def creative_extras(df):
    """Creative: judge sub-scores, lexical diversity, length and truncation per strategy."""
    jp = os.path.join(SCORE_DIR, "creative_judge.csv")
    if not os.path.exists(jp):
        return
    j = pd.read_csv(jp)
    models = set(df[df.task == "creative"].model)
    j = j[j.model.isin(models)]
    dims = ["creativity", "coherence", "fluency", "engagement", "judge_mean"]
    c = df[df.task == "creative"]
    # Average within model first, then across models (each model weighted equally).
    sub = j.groupby(["model", "strategy"])[dims].mean().groupby("strategy").mean()
    lex = c.groupby(["model", "strategy"])[["distinct2", "n_words", "truncated"]].mean().groupby("strategy").mean()
    out = sub.join(lex).loc[list(STRATEGIES)]
    out["judge_parse_fail_pct"] = j.groupby("strategy")[dims[:4]].apply(lambda x: x.isna().any(axis=1).mean() * 100)
    out.to_csv(os.path.join(OUT, "creative_breakdown.csv"))
    print(out.round(3).to_string())
    # Per-model relative change of judge score vs greedy, best sampled strategy per model.
    pm = j.groupby(["model", "strategy"]).judge_mean.mean().unstack()
    rel = pm.sub(pm["greedy"], axis=0).div(pm["greedy"], axis=0) * 100
    rel.round(2).to_csv(os.path.join(OUT, "creative_rel_change_per_model.csv"))
    print(rel.round(2).to_string())


def make_figures(pm, tau, sd):
    tasks = [t for t in TASKS if t in pm.task.unique()]
    strat = list(STRATEGIES)
    colors = ["#1f3b73", "#4878cf", "#6acc65", "#d65f5f", "#b47cc7", "#c4ad66"]
    # Fig 1: mean delta vs greedy (relative %) per task x strategy, with per-model points.
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4), sharey=False)
    for ax, t in zip(np.atleast_1d(axes), tasks):
        x = pm[(pm.task == t) & (pm.strategy != "greedy")]
        for i, s in enumerate(SAMPLED):
            v = x[x.strategy == s].delta.values
            ax.bar(i, v.mean(), color=colors[i + 1], alpha=0.8)
            ax.scatter(np.full(len(v), i) + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color="k", zorder=3)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(len(SAMPLED))); ax.set_xticklabels(SAMPLED, rotation=35)
        ax.set_title(t)
        ax.set_ylabel("Δ vs greedy (judge pts, 1–10)" if t == "creative" else "Δ vs greedy (acc. pts)")
    fig.suptitle("Score change vs greedy decoding (bar = mean over models, dots = individual models)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "delta_vs_greedy.png"), dpi=150); plt.close(fig)

    # Fig 2: per-model score by strategy (lines).
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4))
    for ax, t in zip(np.atleast_1d(axes), tasks):
        x = pm[pm.task == t].pivot(index="model", columns="strategy", values="score")[strat]
        for m in x.index:
            ax.plot(range(len(strat)), x.loc[m].values, marker="o", ms=3, lw=1, label=m)
        ax.set_xticks(range(len(strat))); ax.set_xticklabels(strat, rotation=35); ax.set_title(t)
    np.atleast_1d(axes)[-1].legend(fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "per_model_by_strategy.png"), dpi=150); plt.close(fig)

    # Fig 3: Kendall tau vs greedy with bootstrap CIs.
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.8 / len(SAMPLED)
    for i, s in enumerate(SAMPLED):
        x = tau[(tau.strategy == s)].set_index("task").loc[tasks]
        pos = np.arange(len(tasks)) + (i - len(SAMPLED) / 2 + 0.5) * w
        ax.bar(pos, x.tau_vs_greedy, w, color=colors[i + 1], label=s,
               yerr=[x.tau_vs_greedy - x.ci_lo, x.ci_hi - x.tau_vs_greedy], capsize=2)
    ax.axhline(0.9, ls="--", color="k", lw=0.8); ax.axhline(0.7, ls=":", color="k", lw=0.8)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks); ax.set_ylabel("Kendall τ vs greedy ranking")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7, ncol=5, loc="lower center")
    ax.set_title("Model-ranking stability (95% item-bootstrap CI); dashed=0.9, dotted=0.7")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "kendall_tau.png"), dpi=150); plt.close(fig)

    # Fig 4: seed variability per task.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data = [sd[sd.task == t].seed_std.values for t in tasks]
    axes[0].boxplot(data, tick_labels=tasks); axes[0].set_ylabel("std over 3 seeds (score units)")
    axes[0].set_title("Run-level seed std (per model×strategy)")
    data = [sd[sd.task == t].item_flip.values for t in tasks if t != "creative"]
    axes[1].boxplot(data, tick_labels=[t for t in tasks if t != "creative"])
    axes[1].set_ylabel("% items with outcome differing across seeds"); axes[1].set_title("Item-level seed instability")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "seed_variability.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
