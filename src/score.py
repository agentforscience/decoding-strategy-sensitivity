"""Score generations for factual (alias match), reasoning (final number), code (HumanEval
pass@1 via sandboxed subprocess execution) and creative (diversity/degeneration stats; the
LLM-judge score is added by judge.py). Writes results/scores/item_scores.csv.
"""
import glob
import json
import os
import re
import string
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GEN_DIR, SCORE_DIR, load_task  # noqa: E402


def norm(s):
    """TriviaQA-style normalization: lowercase, strip punctuation/articles/whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score_factual(text, item):
    # Use the first line only (models sometimes add explanations despite the instruction).
    first = text.strip().split("\n")[0] if text.strip() else ""
    p = norm(first)
    if not p:
        return 0.0
    for a in item["aliases"]:
        a = norm(a)
        if a and (p == a or re.search(r"\b" + re.escape(a) + r"\b", p)):
            return 1.0
    return 0.0


NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def extract_number(text):
    m = re.findall(r"answer is[:\s]*\**\s*(" + NUM + ")", text, flags=re.I)
    if not m:
        m = re.findall(r"\\boxed\{(" + NUM + r")\}", text)
    if not m:
        m = re.findall(NUM, text)
    if not m:
        return None
    return m[-1].replace("$", "").replace(",", "").rstrip(".")


def score_reasoning(text, item):
    pred = extract_number(text)
    if pred is None:
        return 0.0
    try:
        return float(abs(float(pred) - float(item["answer"])) < 1e-6)
    except ValueError:
        return 0.0


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    m = re.search(r"```(?:python|py)?\s*\n(.*)", text, flags=re.S)  # unterminated block
    return m.group(1) if m else text


def run_code(item, text, timeout=10):
    """Execute prompt + completion + tests in a fresh python process (isolated, time-limited)."""
    prog = item["prompt"] + "\n" + extract_code(text) + "\n\n" + item["test"] + \
        f"\n\ncheck({item['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prog.py")
        with open(f, "w") as fh:
            fh.write(prog)
        try:
            r = subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, timeout=timeout,
                               stdin=subprocess.DEVNULL)
            return float(r.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0.0


def distinct_n(tokens, n):
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def main():
    os.makedirs(SCORE_DIR, exist_ok=True)
    items = {t: {it["id"]: it for it in load_task(t)} for t in ["factual", "reasoning", "code", "creative"]}
    out_path = os.path.join(SCORE_DIR, "item_scores.csv")
    cache = {}
    if os.path.exists(out_path):  # reuse already-executed code results
        old = pd.read_csv(out_path)
        cache = {(r.model, r.task, r.strategy, r.seed, str(r.id)): r.score for r in old.itertuples()}
    rows, code_jobs = [], []
    for f in sorted(glob.glob(os.path.join(GEN_DIR, "*", "*.jsonl"))):
        model = os.path.basename(os.path.dirname(f))
        task, strategy, seed = os.path.basename(f)[:-6].split("__")
        seed = int(seed[1:])
        for line in open(f):
            g = json.loads(line)
            it = items[task][g["id"]]
            row = dict(model=model, task=task, strategy=strategy, seed=seed, id=g["id"],
                       n_tokens=g["n_tokens"], truncated=int(g["finish_reason"] == "length"))
            if task == "factual":
                row["score"] = score_factual(g["text"], it)
            elif task == "reasoning":
                row["score"] = score_reasoning(g["text"], it)
            elif task == "code":
                key = (model, task, strategy, seed, str(g["id"]))
                if key in cache:
                    row["score"] = cache[key]
                else:
                    code_jobs.append((len(rows), it, g["text"]))
            else:
                toks = g["text"].split()
                row["distinct2"] = distinct_n(toks, 2)
                row["n_words"] = len(toks)
                row["score"] = float("nan")  # filled by judge.py
            rows.append(row)
    print(f"{len(rows)} rows; executing {len(code_jobs)} code samples", flush=True)
    with ThreadPoolExecutor(24) as ex:
        res = list(ex.map(lambda j: run_code(j[1], j[2]), code_jobs))
    for (ri, _, _), r in zip(code_jobs, res):
        rows[ri]["score"] = r
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(df.groupby(["task", "strategy"])["score"].mean().unstack().round(3))


if __name__ == "__main__":
    main()
