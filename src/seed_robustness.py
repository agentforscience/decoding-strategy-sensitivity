"""H5 robustness: seed std with factual/reasoning subsampled to the code set size (164 items),
averaged over 200 random subsets, so the domain comparison isn't driven by differing n."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, SCORE_DIR  # noqa: E402

rng = np.random.default_rng(0)
df = pd.read_csv(os.path.join(SCORE_DIR, "item_scores.csv"))
df["id"] = df["id"].astype(str)
df = df[(df.strategy != "greedy") & df.task.isin(["factual", "reasoning", "code"])]
N = df[df.task == "code"].id.nunique()
rows = []
for task in ["factual", "reasoning", "code"]:
    d = df[df.task == task]
    piv = d.pivot_table(index="id", columns=["model", "strategy", "seed"], values="score")
    ids = piv.index.values
    reps = 1 if task == "code" else 200
    stds = []
    for _ in range(reps):
        sub = piv if task == "code" else piv.loc[rng.choice(ids, N, replace=False)]
        run = sub.mean() * 100  # run-level accuracy per (model, strategy, seed)
        stds.append(run.groupby(level=["model", "strategy"]).std(ddof=1).mean())
    rows.append(dict(task=task, n_items=N, mean_seed_std_pts=np.mean(stds),
                     ci_lo=np.percentile(stds, 2.5), ci_hi=np.percentile(stds, 97.5)))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROOT, "results", "analysis", "seed_std_matched_n.csv"), index=False)
print(out.round(3).to_string())
