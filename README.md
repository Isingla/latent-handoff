# Latent handoff between frozen language models — measured

Can one AI model hand its work to another **without writing it in text**, by passing hidden states through a small
trained bridge? If so, what does that channel actually carry — and where does it break?

One file runs every experiment: [`latent_handoff_experiments.py`](latent_handoff_experiments.py).

```bash
pip install torch transformers accelerate
python latent_handoff_experiments.py --sender Qwen/Qwen2.5-0.5B-Instruct --receiver HuggingFaceTB/SmolLM2-1.7B-Instruct
# ~90 min on one A40. Smoke run (~15 min): add --quick
```

Both models stay **frozen**. Only bridges (~5–30M params) train. Numbers below are from one run of the file above
(`results/full_run.log` + `results/tree_run.log`, ~100 min on an A40), one seed, synthetic tasks, small open models — directions, not production numbers.

## What we found

| # | claim | number (Qwen2.5-0.5B → SmolLM2-1.7B, different families and tokenizers) |
|---|---|---|
| 1 | The handoff works across model families | plan fidelity **0.81–0.82**; same-model 0.86 |
| 2 | It is fast | **17 ms per hop** vs 1.6 s to decode + re-read the plan as text (96×). Full 3-specialist tree: 121 ms vs 3.3 s (28×) |
| 3 | It does not decay | **flat through 8 text-free hops** on the scorecard (hop 8/hop 1 = 0.992); flat through 16 in the overnight chain (0.813 → 0.816) |
| 4 | It leaks what the bridge was trained to carry | a fidelity-only bridge makes the receiver **abandon its own output instructions** (format-following 0.00 vs 1.00 for text). Neither smaller K nor latent position fixes it. The training cue is not the cause (text + cue: 1.00) |
| 5 | A receiver-aware objective fixes it | fidelity + λ·(receiver's own output loss), λ=1: **format-following 0.00 → 1.00 at −0.033 fidelity** |
| 6 | Fan-in works; our combiner is too simple | 3 specialists → one *linear* merge → 1 commander. In 2 of 3 runs each branch's contribution is traceable (drop *runbook*: format 1.00 → 0.00; drop *diagnosis*: severity −0.05; drop *escalation*: nothing). In the 3rd the merge latched onto one branch and ignored the other two. This is an engineering shortcoming of a one-layer combiner, not a property of the channel — swap in an attention merge before drawing conclusions about fan-in |
| 7 | The tree fits in one voice turn | with a **400 ms** turn budget (what a voice assistant has before the pause sounds broken), the text tree reaches a decision in **0%** of turns (3.5 s median); the latent tree in **100%** (123 ms median, 126 ms p90), decisions within ±0.1 of the text tree, runbook format 1.00. Caveat: the text baseline re-decodes an intermediate plan at every hop; where agents can forward the raw transcript instead, the gap shrinks on short inputs |
| 8 | The deadline cliff | under a **100 ms** deadline the text-coordinated team misses **100%** of auctions (1.6 s median); the latent team misses none (72 ms median). With a 3 s deadline, text recovers — the loss is purely time |

**Null result, reported as such:** a frozen-backbone typed decider (severity / bid-or-pass / value) did **not** learn
the decision from *any* channel, text included, at this scale — it landed on the prior in every lane, in both the
incident tree (severity 0.385 text and latent) and the auction (every LLM lane below a constant bid; the latent
bidder never wins). The channel is proven; a decision layer on top of it with frozen weights is not. Next step:
LoRA on the decider, or real features.

**Where this matters (and where it does not).** The channel helps only where LLM-sized models are already in the
loop and the text round trip between them is the bottleneck: multi-agent pipelines (every supervisor→worker hop
costs ~1–2 s and a few hundred tokens; 5–20 hops per task), and voice agents (a ~300–500 ms turn budget that a
text handoff to a sub-agent blows). Real-time bidding is *not* such a place — real bidders are ms-scale tree models
at millions of requests/second, and even the latent team's ~80 ms would not survive network overhead. Stage 3 keeps
the auction because the cliff is a clean result; the market is not the point.

**The one-sentence version:** the channel is fast, faithful, and durable; it carries whatever the bridge's training
signal rewards — content, format, role — so train the bridge on what the *receiver* needs, not only on what the
sender said.

## The scorecard

`latentcheck` (stage 1) runs six probes on any sender → receiver pair and writes `out/report.html`:

```
fidelity     0.820  pass      one-hop plan reconstruction
decay        0.992  pass      fidelity at hop 8 / hop 1, no text between hops
format_keep  0.00   ATTENTION receiver still follows its own instructions? (text: 1.00)
cue_indep    0.00   pass      the training cue is innocent; the leak is in the latents
fix          1.00   pass      format restored by the mixed objective; fidelity cost 0.033
latency      17 ms/hop vs 1644 ms text decode  (96x)
```

Run it on your own pair before turning on latent handoff (`out/report.html` is written by the run; the logs in `results/` are from ours). "ATTENTION" means: works, but not as trained by default.

## Stages

1. **latentcheck** — fidelity, decay, format_keep, cue control, fix, latency. Synthetic planning task
   (supervisor plan → executor brief).
2. **tree** — incident response. One Watcher broadcasts hidden states to three specialist bridges
   (diagnosis / runbook / escalation); a per-slot linear merge feeds the Commander. Text tree vs latent tree;
   leave-one-branch-out.
   Ends with the **voice-turn check**: per-incident time to Commander decision against `--voice-budget-ms`
   (default 400) for both trees, plus whether accuracy and output format survive.
3. **auction** — real-time bidding. Three specialists each see one disjoint slice (user / context / history;
   informativeness 55/30/15 by construction) and hand off to a Commander bidder under a deadline. Second-price
   against three noisy competitors; timeout = pass. Blind / text / latent lanes, per-auction wall-time billing.

## What is synthetic (all of it, labeled)

Planning states, incident alert storms, and auctions are generated. Auction features follow iPinYou's marginals;
hidden CTR is logistic over the slices; clearing prices are second-price against a synthetic field. The public
iPinYou CSV export lacks the paying-price column, so no real-auction replay was possible.

## Related work

Interlat (ACL 2026) and LatentMAS (ICML 2026) communicate between agents in latent space; Cache-to-Cache (ICLR 2026)
fuses KV-caches across frozen heterogeneous models; Coconut (Meta, 2024) reasons in latent space within one model;
Pham et al. (2023) pass embeddings between debating agents. None report decay over hops, what else the channel
carries, or a per-pair test. That is what this repo adds.

## Honest limits

One seed per stage (the tree stage was run three times; see claim 6). Synthetic tasks. 0.5B and 1.7B models. Bridges trained for 300–1,500 steps. The decision-layer null
may be a capacity limit of a linear head on a frozen 1.7B backbone rather than a property of the channel.

## Repo layout

```
latent_handoff_experiments.py   everything: data, bridges, three stages, scorecard, summary
results/full_run.log            stages 1–3, one run (tree stage crashed after its results on a flag typo, re-run below)
results/tree_run.log            stage 2 re-run with the voice-turn check
LICENSE                         MIT
```

## Author

Ishaan Singla — CS, Michigan State University. Built and run alone on a rented A40 over about two weeks.
