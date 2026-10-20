"""Generate outputs for one model across all tasks x strategies x seeds with vLLM.

Usage: python src/generate.py --model qwen2.5-0.5b --gpu-gb 6
Resumable: (task, strategy, seed) files that already exist are skipped.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MAX_TOKENS, MODELS, STRATEGIES, TASKS, gen_path, load_task, make_prompt, runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu-gb", type=float, default=None)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=None, help="debug: items per task")
    args = ap.parse_args()

    repo, need_gb = MODELS[args.model]
    gb = args.gpu_gb or need_gb
    todo = [(t, s, sd) for t in args.tasks.split(",") for s, sd in runs()
            if not os.path.exists(gen_path(args.model, t, s, sd))]
    if not todo:
        print("nothing to do"); return

    import torch
    from vllm import LLM, SamplingParams

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    # Shared GPU: take what is actually free right now (up to the budget), but bail out if that
    # is far below the budget (too little KV cache => extremely slow); the scheduler retries.
    free = torch.cuda.mem_get_info()[0] / 2**30
    gb = min(gb, free - 1.2)  # margin for this process's CUDA context + jitter
    if gb < 0.75 * (args.gpu_gb or need_gb):
        sys.exit(f"only {free:.1f} GB free, need ~{need_gb} GB")
    print(f"using {gb:.1f} GB of {free:.1f} GB free", flush=True)
    llm = LLM(model=repo, gpu_memory_utilization=gb / total, max_model_len=2048,
              max_num_seqs=128, max_num_batched_tokens=4096, seed=0, trust_remote_code=False)
    tok = llm.get_tokenizer()

    for task in args.tasks.split(","):
        items = load_task(task)[: args.limit]
        prompts = []
        for it in items:
            msgs = [{"role": "user", "content": make_prompt(task, it)}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        combos = [(s, sd) for s, sd in runs() if not os.path.exists(gen_path(args.model, task, s, sd))]
        if not combos:
            continue
        # One big batched call covering all (strategy, seed) combos for this task.
        all_prompts, all_sp, keys = [], [], []
        for s, sd in combos:
            for i, p in enumerate(prompts):
                kw = dict(STRATEGIES[s])
                if s != "greedy":
                    kw["seed"] = 100_000 * sd + i  # per-request seed => reproducible regardless of batching
                all_prompts.append(p)
                all_sp.append(SamplingParams(max_tokens=MAX_TOKENS[task], **kw))
                keys.append((s, sd, i))
        t0 = time.time()
        outs = llm.generate(all_prompts, all_sp, use_tqdm=False)
        print(f"[{args.model}] {task}: {len(outs)} gens in {time.time()-t0:.0f}s", flush=True)
        buf = {}
        for (s, sd, i), o in zip(keys, outs):
            c = o.outputs[0]
            buf.setdefault((s, sd), []).append({
                "id": items[i]["id"], "text": c.text, "n_tokens": len(c.token_ids),
                "finish_reason": c.finish_reason})
        for (s, sd), rows in buf.items():
            p = gen_path(args.model, task, s, sd)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p + ".tmp", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            os.replace(p + ".tmp", p)
    print("DONE", args.model)


if __name__ == "__main__":
    main()
