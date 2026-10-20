"""Shared configuration: models, decoding strategies, task prompts, data loading."""
import glob
import json
import os
import random

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB = os.environ.get("HF_HUB_CACHE", "/data/transformers/hub")
DATA_DIR = os.path.join(ROOT, "data")
GEN_DIR = os.path.join(ROOT, "results", "gen")
SCORE_DIR = os.path.join(ROOT, "results", "scores")
SEED = 42

# key -> (HF repo id, approx GPU GB needed incl. KV cache)
MODELS = {
    "qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", 4),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", 7),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", 11),
    "qwen2.5-7b-awq": ("Qwen/Qwen2.5-7B-Instruct-AWQ", 11),
    "qwen3-4b-2507": ("Qwen/Qwen3-4B-Instruct-2507", 14),
    "llama3.1-8b-awq": ("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", 11),
    "gemma2-9b-awq": ("hugging-quants/gemma-2-9b-it-AWQ-INT4", 15),
    "mistral-7b-awq": ("solidrust/Mistral-7B-Instruct-v0.3-AWQ", 10),
    "falcon3-7b-awq": ("tiiuae/Falcon3-7B-Instruct-AWQ", 11),
    "phi3-mini-gptq": ("kaitchup/Phi-3-mini-4k-instruct-gptq-4bit", 11),
    "olmo3-7b-gptq": ("kaitchup/Olmo-3-7B-Instruct-gptq-w4a16-g128", 14),
}
# Judge: not among evaluated models; 4-bit so it fits the small GPU slices on the shared host.
JUDGE_MODEL = "Qwen/Qwen3-8B-AWQ"

# Decoding strategies. Sampled strategies are run with SEEDS; greedy once.
STRATEGIES = {
    "greedy": dict(temperature=0.0),
    "t0.7": dict(temperature=0.7),
    "t1.0": dict(temperature=1.0),
    "topp0.9": dict(temperature=1.0, top_p=0.9),
    "topk40": dict(temperature=1.0, top_k=40),
    "minp0.1": dict(temperature=1.0, min_p=0.1),
}
SEEDS = [0, 1, 2]
TASKS = ["factual", "reasoning", "code", "creative"]
MAX_TOKENS = {"factual": 48, "reasoning": 640, "code": 640, "creative": 400}
N_ITEMS = {"factual": 400, "reasoning": 400, "code": 164, "creative": 100}


def runs():
    """All (strategy, seed) pairs."""
    out = [("greedy", 0)]
    for s in STRATEGIES:
        if s != "greedy":
            out += [(s, sd) for sd in SEEDS]
    return out


def _pq(pattern):
    return pd.read_parquet(glob.glob(os.path.join(HUB, pattern))[0])


def build_data():
    """Sample evaluation items once and freeze them in data/*.jsonl."""
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = random.Random(SEED)
    tq = _pq("datasets--mandarjoshi--trivia_qa/snapshots/*/rc.nocontext/validation-*.parquet")
    idx = rng.sample(range(len(tq)), N_ITEMS["factual"])
    fact = [{"id": tq.iloc[i]["question_id"], "question": tq.iloc[i]["question"],
             "aliases": list(tq.iloc[i]["answer"]["normalized_aliases"]) + [tq.iloc[i]["answer"]["normalized_value"]]}
            for i in idx]
    gs = _pq("datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")
    idx = rng.sample(range(len(gs)), N_ITEMS["reasoning"])
    reas = [{"id": f"gsm8k-{i}", "question": gs.iloc[i]["question"],
             "answer": gs.iloc[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in idx]
    he = _pq("datasets--openai--openai_humaneval/snapshots/*/openai_humaneval/test-*.parquet")
    code = [{"id": r.task_id, "prompt": r.prompt, "test": r.test, "entry_point": r.entry_point}
            for r in he.itertuples()]
    roc = _pq("datasets--hamishivi--ROCStories/snapshots/*/data/test-*.parquet")
    col = "prompt"  # first sentence of the story
    idx = rng.sample(range(len(roc)), N_ITEMS["creative"])
    crea = [{"id": f"roc-{i}", "first_sentence": str(roc.iloc[i][col])} for i in idx]
    for name, items in [("factual", fact), ("reasoning", reas), ("code", code), ("creative", crea)]:
        with open(os.path.join(DATA_DIR, f"{name}.jsonl"), "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(name, len(items))


def load_task(task):
    with open(os.path.join(DATA_DIR, f"{task}.jsonl")) as f:
        return [json.loads(l) for l in f]


def make_prompt(task, item):
    """Zero-shot user message for each task (identical across models and strategies)."""
    if task == "factual":
        return ("Answer the following trivia question. Reply with only the answer "
                "(a few words), no explanation.\n\nQuestion: " + item["question"] + "\nAnswer:")
    if task == "reasoning":
        return ("Solve the following math word problem. Think step by step, then write the final "
                "answer on the last line in the form \"The answer is N\".\n\n" + item["question"])
    if task == "code":
        return ("Complete the following Python function. Return the complete function "
                "(signature and body, plus any imports it needs) in a single ```python code block.\n\n"
                "```python\n" + item["prompt"] + "```")
    if task == "creative":
        return ("Write a short, original story of about 150 words that begins with the sentence "
                "below. Make it vivid, surprising and engaging. Output only the story.\n\n"
                "First sentence: " + item["first_sentence"])
    raise ValueError(task)


def gen_path(model, task, strategy, seed):
    return os.path.join(GEN_DIR, model, f"{task}__{strategy}__s{seed}.jsonl")


if __name__ == "__main__":
    build_data()
