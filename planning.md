# Planning: Decoding Strategy Sensitivity Across Task Domains

## Motivation & Novelty Assessment

### Why This Research Matters
Benchmark numbers for LLMs are routinely reported under a single, often unstated, decoding
configuration (greedy for leaderboards, T≈0.7 for chat). If a task family's scores or model
*rankings* depend on the decoding strategy, then cross-paper comparisons and leaderboard
conclusions for that family are fragile. Practitioners also need to know which tasks tolerate
sampling (and which sampler to use) without losing accuracy.

### Gap in Existing Work
Note: the resource-finder stage was skipped for this workspace (no `literature_review.md`,
`papers/`, `datasets/`, `code/`), so this is based on prior knowledge. Prior work studies
decoding mostly one task family at a time: Holtzman et al. 2020 (nucleus sampling, open-ended
text), Shi et al. 2024 "A Thorough Examination of Decoding Methods in the Era of LLMs"
(many tasks, but few models and no ranking-stability analysis), Renze & Guven 2024 (temperature
has little effect on MCQA problem solving), Chen et al. 2021 (Codex: temperature vs pass@k),
Nguyen et al. 2024 (min-p sampling). What has not been looked at much is a **single controlled
grid** of models × task domains × decoding strategies × seeds that measures (a) score
sensitivity per domain, (b) how stable the **model rankings** are across strategies
(Kendall τ), and (c) how much each domain varies across seeds.

### Our Novel Contribution
A controlled 11-model × 4-domain × 6-strategy × 3-seed study that tests the five quantitative
claims in the hypothesis, especially the ranking-stability claim (τ>0.9 factual vs τ<0.7
reasoning), which needs many models to test.

### Experiment Justification
- Exp 1 (Factual QA, TriviaQA): tests "factual drops <2% across strategies" and τ>0.9.
- Exp 2 (Reasoning, GSM8K CoT): tests the 5–12% greedy-vs-T=1.0 gap, top-p recovery, τ<0.7.
- Exp 3 (Code, HumanEval pass@1): tests "code shows highest seed variance under sampling".
- Exp 4 (Creative, story writing + LLM-judge proxy): tests "sampling improves creative quality 10–15%".
- Exp 5 (Cross-cutting ranking analysis): Kendall τ between the greedy ranking and each
  strategy's ranking, per domain, with bootstrap CIs.

## Research Question
How sensitive are LLM task scores and model rankings to the decoding strategy, and does that
sensitivity differ between factual QA, multi-step reasoning, code synthesis and creative writing?

## Hypothesis Decomposition
- H1: For every model, |acc(strategy) − acc(greedy)| < 2 points on factual QA (absolute; we also report relative).
- H2: GSM8K accuracy at T=1.0 is 5–12 points below greedy; top-p=0.9 (at T=1.0) recovers most (>50%) of that gap.
- H3: Creative quality (LLM-judge score) under sampling is 10–15% higher (relative) than greedy.
- H4: Kendall τ(greedy ranking, strategy ranking) > 0.9 for factual; < 0.7 for reasoning.
- H5: Std across seeds (per model, per strategy) is largest for code among the four domains.

## Proposed Methodology

### Approach
No API keys are available on this host, so we use **real open-weight instruction-tuned LLMs
served locally with vLLM** (shared RTX A6000 GPUs). 11 models from 7 families:
Qwen2.5-{0.5B,1.5B,3B}-Instruct, Qwen2.5-7B-Instruct-AWQ, Qwen3-4B-Instruct-2507,
Llama-3.1-8B-Instruct-AWQ, gemma-2-9b-it-AWQ, Mistral-7B-Instruct-v0.3-AWQ,
Falcon3-7B-Instruct-AWQ, Phi-3-mini-4k-instruct-GPTQ, Olmo-3-7B-Instruct-GPTQ.
The wide spread in model quality is realistic for leaderboards. Small models plus local serving
make a full grid (≈ 11×6×3 runs per task) affordable.

Decoding strategies (6):
1. greedy (T=0)
2. T=0.7 (pure temperature)
3. T=1.0 (pure ancestral sampling)
4. top-p=0.9 at T=1.0
5. top-k=40 at T=1.0
6. min-p=0.1 at T=1.0

Sampled strategies use seeds {0,1,2}. Greedy is run once (deterministic, modulo batch nondeterminism).

### Experimental Steps
1. Data: TriviaQA rc.nocontext validation (n=400 random), GSM8K test (n=400 random),
   HumanEval (all 164), creative: 100 story prompts built from ROCStories first sentences.
2. Generation with vLLM, chat template, zero-shot prompts (in `src/prompts.py`).
3. Scoring: TriviaQA = normalized alias match (EM-or-contains on the short answer);
   GSM8K = last number vs gold; HumanEval = pass@1 by executing the tests in a subprocess with a timeout;
   Creative = Qwen2.5-14B-Instruct (not one of the evaluated models) as judge with a 1–10 rubric
   (creativity, coherence, fluency, engagement; we report the mean), plus distinct-n and
   degeneration rate as auxiliary metrics.
4. Analysis: per (model, domain, strategy) mean over seeds; deltas vs greedy; seed std;
   Kendall τ of model rankings; bootstrap CIs over items; paired tests.

### Baselines
Greedy decoding is the reference condition for every comparison. T=0.7 is the common "chat default".

### Evaluation Metrics
Accuracy / pass@1 / judge score (as described above); Δ vs greedy (absolute points and relative %);
seed std; Kendall τ_b.

### Statistical Analysis Plan
- Per-domain mean Δ vs greedy across models: one-sample t-test and Wilcoxon signed-rank over the 11 models; 95% bootstrap CIs.
- Item-level paired comparisons (greedy vs strategy) pooled across models: McNemar/sign test.
- Kendall τ with 95% CI from bootstrap resampling of items (rankings recomputed for each resample).
- Seed variance: compare per-model seed std across domains (Friedman/Wilcoxon), and also
  normalized by the binomial std to control for differing n and base rates.
- α = 0.05, Holm correction within each family of tests.

## Expected Outcomes
Support: factual |Δ|<2; reasoning gap 5–12 with top-p recovery; creative +10–15%; τ_fact>0.9, τ_reason<0.7;
code has the largest seed std. Any domain falling outside these ranges counts as evidence against that sub-claim.

## Timeline and Milestones
Setup 30 min; pipeline test 30 min; generation ≈ 2–3 h (GPU contention dependent); judge 30–60 min;
analysis 45 min; documentation 45 min.

## Potential Challenges
- Shared GPUs: other jobs take memory → small gpu_memory_utilization, retry loop, one model at a time.
- Small models can degenerate at T=1.0 (very long outputs) → max_tokens caps; report degeneration.
- LLM-judge ≠ human judgment → state that it's a proxy; check judge self-consistency; report diversity metrics too.
- Batch nondeterminism in "greedy" → acknowledged.

## Success Criteria
Full grid completed for ≥8 models in all 4 domains, every sub-hypothesis given a supported/refuted
verdict with CIs, and REPORT.md written.

## Deviations from Plan (recorded after execution)
- Creative judge: Qwen2.5-14B-Instruct did not fit the shared GPUs, so Qwen3-8B-AWQ (non-thinking mode; not an evaluated model) was used instead.
- GPU contention (other users' jobs) forced adaptive per-model memory budgets and a retry scheduler; Phi-3 ran on a separate 4.8 GB slice. None of this affects outputs, because per-request seeds make results independent of batching.
- Added a matched-n (164-item) seed-variance robustness check for H5, because code has fewer items than the other domains.
