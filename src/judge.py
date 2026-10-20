"""LLM-as-judge proxy for creative quality. A held-out judge model (Qwen3-8B-AWQ in non-thinking mode,
not one of the evaluated models) rates each story 1-10 on creativity, coherence, fluency and
engagement, blind to model and decoding strategy. Greedy judge decoding.

Writes results/scores/creative_judge.csv (model, strategy, seed, id, 4 sub-scores, mean).
"""
import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GEN_DIR, JUDGE_MODEL, SCORE_DIR, load_task  # noqa: E402

DIMS = ["creativity", "coherence", "fluency", "engagement"]
JUDGE_PROMPT = """You are an expert creative-writing teacher grading short stories written from a given first sentence.
Rate the story on four criteria, each an integer from 1 (very poor) to 10 (excellent):
- creativity: originality of ideas, surprising but sensible developments, avoids cliches
- coherence: the plot is logically consistent and follows from the first sentence; has a clear arc
- fluency: grammatical, natural, well-formed prose without repetition, gibberish or broken text
- engagement: how interesting and enjoyable it is to read

Be strict and use the full scale. A story that is truncated, degenerate, repetitive or off-task should get low scores.

First sentence: {first}

Story:
\"\"\"
{story}
\"\"\"

Respond with only a JSON object like {{"creativity": x, "coherence": x, "fluency": x, "engagement": x}}."""


def parse(text):
    out = {}
    for d in DIMS:
        m = re.search(r'"?' + d + r'"?\s*:\s*(\d+(?:\.\d+)?)', text, flags=re.I)
        out[d] = min(10.0, max(1.0, float(m.group(1)))) if m else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-gb", type=float, default=11)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    import torch
    from vllm import LLM, SamplingParams

    firsts = {it["id"]: it["first_sentence"] for it in load_task("creative")}
    rows, prompts = [], []
    for f in sorted(glob.glob(os.path.join(GEN_DIR, "*", "creative__*.jsonl"))):
        model = os.path.basename(os.path.dirname(f))
        _, strategy, seed = os.path.basename(f)[:-6].split("__")
        for line in open(f):
            g = json.loads(line)
            rows.append(dict(model=model, strategy=strategy, seed=int(seed[1:]), id=g["id"]))
            # Cap story length fed to the judge (outputs are already capped at 400 tokens).
            prompts.append(JUDGE_PROMPT.format(first=firsts[g["id"]], story=g["text"].strip()[:3000]))
    rows, prompts = rows[: args.limit], prompts[: args.limit]
    out_path = os.path.join(SCORE_DIR, "creative_judge.csv" if args.limit is None else "creative_judge_test.csv")
    done = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame(columns=["model", "strategy", "seed", "id"])
    have = set(zip(done.model, done.strategy, done.seed.astype(int), done.id.astype(str)))
    keep = [i for i, r in enumerate(rows) if (r["model"], r["strategy"], r["seed"], r["id"]) not in have]
    rows, prompts = [rows[i] for i in keep], [prompts[i] for i in keep]
    print(f"{len(rows)} stories to judge ({len(have)} cached)", flush=True)
    if not rows:
        return
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    llm = LLM(model=JUDGE_MODEL, gpu_memory_utilization=args.gpu_gb / total, max_model_len=2048,
              max_num_seqs=128, max_num_batched_tokens=4096, seed=0)
    tok = llm.get_tokenizer()
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for p in prompts]
    outs = llm.generate(chats, SamplingParams(temperature=0.0, max_tokens=60))
    for r, o in zip(rows, outs):
        r.update(parse(o.outputs[0].text))
        r["judge_raw"] = o.outputs[0].text
    df = pd.DataFrame(rows)
    df["judge_mean"] = df[DIMS].mean(axis=1)
    df = pd.concat([done, df], ignore_index=True) if len(done) else df
    os.makedirs(SCORE_DIR, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("parse failures:", df[DIMS].isna().any(axis=1).sum())
    print(df.groupby("strategy")["judge_mean"].mean().round(3))


if __name__ == "__main__":
    main()
