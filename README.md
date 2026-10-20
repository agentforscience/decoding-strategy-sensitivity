# Decoding Strategy Sensitivity Across Task Domains

This project runs a controlled grid of **11 open LLMs × 4 domains × 6 decoding strategies × 3 seeds**, about 132k generations served locally with vLLM. The domains are factual QA (TriviaQA), math reasoning (GSM8K), code (HumanEval) and creative writing (ROCStories prompts scored by an LLM judge). It tests how much scores, model rankings and seed variance depend on the decoder. Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings
- **No sampler beat greedy on any verifiable task.** Raw T=1.0 and top-k 40 cost 4–7 points on factual, reasoning and code. Top-p 0.9 and min-p 0.1 recover about half of that loss; min-p was the best sampler overall.
- **Factual QA is *not* decoding-robust** (refutes the "<2% drop" claim). The mean drop is 1.6 points (T=0.7/min-p) to 4.3 points (T=1.0), and up to 8 points for a single model.
- **Reasoning:** the greedy→T=1.0 loss is 5.9 points on average (in the hypothesised 5–12 range), but strong models lose under 1 point. Top-p recovers 50%.
- **Rankings are stable:** Kendall τ vs greedy is 0.93 for factual and 0.85–0.89 for reasoning (the least stable verifiable domain, but not < 0.7). The only τ near 0.7 is creative writing at T=1.0.
- **Creative quality does not improve with sampling** under the LLM-judge proxy: the score changes by 0 to +0.7% (and −8% at T=1.0). Diversity and the creativity sub-score rise, but coherence and fluency fall.
- **Code has the largest seed variance:** seed SD 2.1 points and 27% of items flip between seeds. The margin shrinks with matched test-set size.

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/data/transformers  # local HF cache
python src/common.py          # build frozen eval sets in data/
python src/run_all.py         # GPU scheduler: src/generate.py per model -> results/gen/
python src/score.py           # accuracy / pass@1 (sandboxed execution) -> results/scores/item_scores.csv
python src/judge.py           # creative LLM judge (Qwen3-8B-AWQ) -> results/scores/creative_judge.csv
python src/analyze.py         # stats + figures -> results/analysis/, figures/
python src/seed_robustness.py # matched-n seed-variance check
```
Models and datasets are read from the local HF cache (`/data/transformers/hub`). On a free 48 GB GPU, generation takes about 1–2 h. On the shared host used here it took about 4.5 h.

## Layout
```
planning.md            research plan and motivation
REPORT.md              full report with results
src/common.py          models, strategies, prompts, data sampling
src/generate.py        vLLM generation for one model (all tasks × strategies × seeds)
src/run_all.py         GPU-aware scheduler with retries (shared GPUs)
src/score.py           scoring (TriviaQA alias match, GSM8K number, HumanEval execution)
src/judge.py           LLM-as-judge for creative stories
src/analyze.py         deltas, Wilcoxon/Holm, Kendall tau + bootstrap CIs, seed variance, figures
src/seed_robustness.py matched-n seed-variance check
data/                  frozen evaluation items (jsonl)
results/gen/           raw generations per model/task/strategy/seed
results/scores/        item-level scores and judge outputs
results/analysis/      summary tables (csv/json)
figures/               plots
logs/                  generation/judge/analysis logs
```
