#!/usr/bin/env python3
"""
latent_handoff_experiments.py — every experiment in this project, one file, start to finish.

Question: can AI agents hand work to each other WITHOUT writing it in text (passing hidden states through a small
trained bridge instead), and what does that channel actually carry?  Both models stay FROZEN; only bridges train.

Stages (each prints its own table; a summary prints at the end; JSON + an HTML scorecard land in ./out):
  1. latentcheck  — six probes on one sender->receiver pair:
       fidelity (one-hop plan reconstruction), decay over H text-free hops, format_keep (does the receiver still
       follow ITS OWN instructions after the handoff?), cue_indep (control: is a leak caused by the training cue?),
       fix (a receiver-aware objective restores format; fidelity cost), latency (ms per hop vs text decode).
  2. tree         — fan-out/fan-in on an incident-response task: one Watcher -> 3 specialists -> Commander, text
       tree vs latent tree; leave-one-branch-out shows which branch carries what.
  3. auction      — real-time-bidding demo: 3 specialist senders -> Commander bidder under a 100 ms deadline
       (text memos vs latent handoff), with a 3 s control deadline. Shows the deadline cliff.

Run (GPU recommended; CPU works with --quick):
    pip install torch transformers accelerate
    python latent_handoff_experiments.py --sender Qwen/Qwen2.5-0.5B-Instruct --receiver HuggingFaceTB/SmolLM2-1.7B-Instruct
    python latent_handoff_experiments.py --quick          # small config, ~15 min on a single GPU
    python latent_handoff_experiments.py --stages latentcheck   # one stage only

Honest label on every number: one seed, synthetic tasks, small open models, frozen endpoints, bridges only.
Results we got (A40, Qwen-0.5B -> SmolLM2-1.7B): fidelity 0.81, no decay over 8-16 hops, format_keep 0.00 -> 1.00
with the fix (2.4 pts fidelity), 24 ms/hop vs 1.7 s text; text tree 3.3 s vs latent tree 120 ms; at a 100 ms auction
deadline the text team never bids and the latent team always does. The frozen typed decider did NOT learn value
estimation from any channel at this scale (reported as a null).
"""
import argparse, json, math, os, random, re, sys, time, threading, html
import torch, torch.nn as nn, torch.nn.functional as F

# The sections below were developed as separate modules (qwen_bridge, latentcheck, incident_race, tree_race,
# auction_env, auction_tree). They reference each other through these aliases, which all point at this file.
Q = IR = AE = sys.modules[__name__]
SLOT = "<<LATENT_DIRECTIVES>>"

# ====================================================================================== models, prompts, bridge

# ======================================================================= HF backend (inlined)
class _HF: pass
hfbackend = _HF()
_lock = threading.Lock()
_models = {}


_lock = threading.Lock()
_models = {}


def device():
    d = os.environ.get("HF_DEVICE")
    if d:
        return d
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(name=None, agent="FLYWHEEL"):
    """Return (tokenizer, model) for `name`, cached. Models are frozen and in eval mode."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = name or os.environ.get(f"{agent}_HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    with _lock:
        if name in _models:
            return _models[name]
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float32 if device() == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device()).eval()
        for p in model.parameters():
            p.requires_grad = False
        _models[name] = (tok, model)
        return tok, model


def render(tok, messages):
    """Chat template if the tokenizer has one, else plain concatenation."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\n\nASSISTANT:"


def chat(messages, temperature=0.4, max_tokens=800, agent="FLYWHEEL", model_name=None):
    import torch
    tok, model = load(model_name, agent)
    text = render(tok, messages)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                             temperature=max(temperature, 1e-5), pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

hfbackend.load = load; hfbackend.render = render; hfbackend.chat = chat; hfbackend._models = _models; hfbackend.device = device

# ======================================================================= agent prompts (inlined from the repo)
_TOPICS = {
    "ml-math": "Math: linear algebra & probability",
    "ml-classical": "Classical ML: regression, overfitting, validation",
    "ml-nets": "Neural nets: backprop, activations, optimization",
    "ml-pytorch": "PyTorch fluency: tensors, autograd, training loops",
    "ml-finetune": "Fine-tuning & LoRA",
    "ml-eval": "Evaluation: metrics, harnesses, error analysis",
    "llm-prompt": "Prompt engineering",
    "llm-rag": "RAG: embeddings, chunking, retrieval",
    "llm-tools": "Tool use & function calling",
    "llm-agents": "Agents: ReAct, planning, memory, orchestration",
    "llm-evals": "Evals for LLM systems",
    "llm-prod": "Production: cost, latency, guardrails, red-teaming",
}

PLAN_PROMPT = """You are the supervisor of two agents serving one goal: get {user} hired as an AI engineer.

AGENT 1 (tutor): teaches him ML foundations + LLM engineering. Talking caps at level 2/4;
levels 3-4 require SHIPPED artifacts (repos that run, with measured results, defended orally).
AGENT 2 (brief): emails {user} a morning plan; {user} applies to roles BY HAND and logs each one. No agent ever sends applications or messages.

The flywheel: learn -> ship proof -> applications leading with proof -> market feedback
(replies, interviews) -> feed what the market values back into what to learn next.

Current state (JSON):
{state}

Bottleneck diagnosis: [{bottleneck}] {bottleneck_why}
Outreach tier unlocked by proof: [{tier}] {tier_why}

Issue 3-5 directives for this week. Each directive has:
- kind: learn | build | outreach | stop   (outreach = applications/referrals {user} sends personally)
- title: short imperative
- rationale: one sentence linking it to the bottleneck or a market signal
- action: the concrete next step (a command to run, a person to email, a repo to start)

Rules:
- Respect the tier: at tier 0, NO outreach directives. At higher tiers, applications must lead with a specific shipped artifact.
- If market signals point at a tutor topic where his level is low, that becomes a learn/build directive — this is the feedback loop.
- Include a "stop" directive if he is doing something low-leverage (e.g. outreach with no proof, studying a topic the market ignores).
- Be concrete and terse. No generic advice.

Reply as a JSON array only, no prose:
[{{"kind": "...", "title": "...", "rationale": "...", "action": "..."}}]"""

BRIEF_PROMPT = """You write a short morning brief for {user}, a CS undergrad who wants to be hired as an
AI/ML engineer (backend/infra SWE or ML/AI internships, Summer 2027). Today is {today}.
{user} applies to jobs BY HAND; you only plan. Never draft or send anything on their behalf.

STATE (JSON): {state}

Write the brief in Markdown, under 350 words, with exactly these sections:

## 1. Learn (25 min)
One tutor command to run today (`learn <topic>` on the weakest topic, or `review` if reviews are due) and the one idea to nail.

## 2. Build (60 min)
The single concrete step toward the next open gate ({next_gate}). Name the file/function/experiment. If an artifact is ready, say `ship <topic> <path>`.

## 3. Apply ({n_apply} by hand)
{apply_rule}
List {n_apply} specific internship postings for Summer 2027 found via web search TODAY (company — role — URL — one line on why it fits and which shipped artifact to lead with). Skip companies in already_applied. If web search is unavailable, give {n_apply} target companies + where to look instead, and say so.
Then: follow-ups due (stale applications > 10 days: one line each).

## 4. Supervisor says
Bottleneck + tier in one line, then the open flywheel directives as a checklist (max 4).

Rules: terse, imperative, no motivational filler. Every line must be actionable today."""

def bottleneck(state):
    """Return (code, headline, explanation) — the one thing constraining hiring."""
    n_art = state["n_artifacts"]
    contacted, replied = state["contacted"], state["replied"]
    rr = state["reply_rate"]
    n_iv, n_off = state["n_interviews"], state["n_offers"]
    if n_off > 0:
        return ("WON", "Offer on the table.",
                "The flywheel worked. Decide: accept, or keep it spinning for leverage.")
    if n_art == 0:
        return ("PROOF",
                "No proof yet — applying at volume is premature.",
                "Every application sent without a shipped artifact burns a lead. "
                "The highest-leverage move is shipping artifact #1, not application #50.")
    if n_iv > 0:
        return ("CLOSING",
                "Interviews happening, no offer yet.",
                "Bottleneck is interview performance. Log feedback from every round "
                "as market signals; the tutor should drill exactly what interviewers probe.")
    if replied > 0:
        return ("CONVERSION",
                "Replies but no interviews.",
                "Your proof gets attention but doesn't convert. Sharpen the defense: "
                "can you explain each artifact's key decision and numbers cold?")
    if contacted >= 10 and rr < 0.10:
        return ("TARGETING",
                "Proof exists, but <10% reply rate.",
                "Bottleneck is targeting or message, not skill. Fewer, sharper, "
                "stunt-led messages to better-matched companies.")
    if contacted == 0:
        return ("ACTIVATION",
                f"{n_art} artifact(s) shipped, zero applications sent.",
                "Proof is sitting idle. The flywheel only spins when proof meets market.")
    return ("MOMENTUM",
            "Pipeline is moving.",
            "Keep shipping and keep contacting. Watch for which artifacts get replies.")


def tier(state):
    """Outreach tier unlocked by shipped proof."""
    n = state["n_artifacts"]
    if n >= 4:
        return (3, "Tier 3 — Interview prep + referrals. Applications are inbound-assisted now; "
                   "optimize for converting the pipeline, not widening it.")
    if n >= 2:
        return (2, "Tier 2 — ~10 hand-sent applications/week, including bigger companies. "
                   "Lead every application with the Working Demo of your best artifact.")
    if n == 1:
        return (1, "Tier 1 — ~5 hand-sent applications/week to startups. "
                   "Every application leads with your one artifact and its measured result.")
    return (0, "Tier 0 — Build in private. Minimal applications: with zero proof, "
               "every message trains the market to ignore you.")

# ======================================================================= experiment

SLOT = "<<LATENT_DIRECTIVES>>"
TOPICS = list(_TOPICS)


# ---------------------------------------------------------------- synthetic states

def synth_state(rng):
    """A plausible flywheel state, in the exact shape compute_state() produces."""
    levels = {t: rng.choice([0, 1, 1, 2, 2, 3]) for t in TOPICS}
    arts = [{"topic": t, "location": f"github.com/me/{t}-demo", "level": 3}
            for t in rng.sample(TOPICS, rng.choice([0, 0, 1, 1, 2, 3]))]
    contacted = rng.choice([0, 0, 3, 8, 15, 30]); replied = min(contacted, rng.choice([0, 0, 1, 2, 4]))
    sig_pool = [("recruiter asked about LLM evals", "llm-evals"), ("posting wants vLLM latency work", "llm-prod"),
                ("interviewer probed RAG chunking", "llm-rag"), ("asked to explain LoRA tradeoffs", "ml-finetune"),
                ("take-home used tool calling", "llm-tools")]
    sigs = [{"text": s, "source": "reply", "topic": t, "weight": rng.choice([1, 2])} for s, t in rng.sample(sig_pool, rng.randint(0, 3))]
    return {"tutor_db_ok": True, "outreach_db_ok": True, "levels": levels, "artifacts_accepted": arts,
            "n_artifacts": len(arts), "sessions_7d": rng.randint(0, 7),
            "funnel": {"applied": contacted, "replied": replied}, "contacted": contacted, "replied": replied,
            "reply_rate": round(replied / contacted, 3) if contacted else 0.0,
            "recent_replies": [], "n_interviews": rng.choice([0, 0, 0, 1]), "n_offers": 0,
            "signals": sigs, "open_directives": []}


def plan_prompt(state):
    bn, why = bottleneck(state)[0], bottleneck(state)[2]
    t, twhy = tier(state)
    return PLAN_PROMPT.format(user="Ishaan", state=json.dumps(state, indent=1)[:6000],
                                     bottleneck=bn, bottleneck_why=why, tier=t, tier_why=twhy)


def brief_prompt(state, directives_text):
    """BRIEF_PROMPT exactly as brief.brain builds it; directives come from the flywheel."""
    bn = bottleneck(state); t = tier(state)
    bstate = {"tutor": {"levels": state["levels"], "artifacts": state["artifacts_accepted"], "reviews_due": 0,
                        "gates": [], "weakest": min(state["levels"], key=state["levels"].get), "next_gate": None},
              "flywheel": {"bottleneck": {"name": bn[0], "why": bn[2]}, "tier": {"n": t[0], "rule": t[1]},
                           "directives": directives_text, "signals": state["signals"]},
              "applications": {"funnel": state["funnel"], "applied_today": 0, "recent": [], "stale": [], "already_applied": []}}
    return BRIEF_PROMPT.format(user="Ishaan", today="Sunday 2026-10-04", state=json.dumps(bstate, default=str)[:9000],
                                      next_gate="first shipped artifact", n_apply=1,
                                      apply_rule="Spend most of today on Build.")


# ---------------------------------------------------------------- models

class Bridge(nn.Module):
    """K learned queries cross-attend over A's last hidden states -> K vectors in B's embedding space."""
    def __init__(self, d_a, d_b, k=32, nh=8):
        super().__init__()
        d = max(d_a, 256)
        self.inp = nn.Linear(d_a, d)
        self.q = nn.Parameter(torch.randn(k, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d_b))
    def forward(self, h_a, pad_mask=None):
        h = self.inp(h_a)
        q = self.q.unsqueeze(0).expand(h.size(0), -1, -1)
        o, _ = self.attn(q, h, h, key_padding_mask=pad_mask)
        return self.out(o)


def embed(model):
    return model.get_input_embeddings()


@torch.no_grad()
def a_hidden(tok_a, model_a, prompts):
    enc = tok_a([hfbackend.render(tok_a, [{"role": "user", "content": p}]) for p in prompts],
                return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model_a.device)
    out = model_a(**enc, output_hidden_states=True)
    return out.hidden_states[-1].float(), enc["attention_mask"] == 0


def b_inputs(tok_b, model_b, lat, state, suffix_text):
    """embed(prefix) ++ latents ++ embed(suffix) where prefix/suffix split BRIEF_PROMPT at the directives slot."""
    full = hfbackend.render(tok_b, [{"role": "user", "content": brief_prompt(state, SLOT)}]) + suffix_text
    pre, post = full.split(SLOT)
    pre_ids = tok_b(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(model_b.device)
    post_ids = tok_b(post, return_tensors="pt", add_special_tokens=False).input_ids.to(model_b.device)
    E = embed(model_b)
    x = torch.cat([E(pre_ids), lat.to(E.weight.dtype), E(post_ids)], 1)
    return x, pre_ids.size(1), post_ids


def train_bridge(bridge, tok_a, model_a, tok_b, model_b, data, steps, lr=1e-3, log=print):
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr)
    t0 = time.time()
    for i in range(steps):
        state, directives = random.choice(data)
        hA, mask = a_hidden(tok_a, model_a, [plan_prompt(state)])
        lat = bridge(hA, mask)
        # teacher-force: after the latents + a short cue, B must emit the directive text
        cue = "\nThe supervisor's directives are:\n"
        d_ids = tok_b(directives, return_tensors="pt", add_special_tokens=False).input_ids.to(model_b.device)
        x, n_pre, post_ids = b_inputs(tok_b, model_b, lat, state, cue)
        E = embed(model_b)
        x = torch.cat([x, E(d_ids)], 1)
        logits = model_b(inputs_embeds=x).logits.float()
        n_ctx = x.size(1) - d_ids.size(1)
        loss = F.cross_entropy(logits[0, n_ctx - 1:-1], d_ids[0])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0); opt.step()
        if i % max(1, steps // 10) == 0 or i == steps - 1:
            log(f"  [bridge] step {i:4d}  directive-token CE {loss.item():.3f}  ({time.time()-t0:.0f}s)")


@torch.no_grad()
def eval_bridge(bridge, tok_a, model_a, tok_b, model_b, data, gen_tokens, log=print):
    """Directive-token accuracy (teacher-forced) from latents, plus brief generation on both paths."""
    bridge.eval(); accs = []; hits_t = hits_l = 0; n_titles = 0; lat_t = lat_l = 0.0
    E = embed(model_b)
    for state, directives in data:
        # --- latent path
        t0 = time.perf_counter()
        hA, mask = a_hidden(tok_a, model_a, [plan_prompt(state)])
        lat = bridge(hA, mask)
        x, _, _ = b_inputs(tok_b, model_b, lat, state, "")
        out_l = model_b.generate(inputs_embeds=x, max_new_tokens=gen_tokens, do_sample=False, pad_token_id=tok_b.pad_token_id)
        lat_l += time.perf_counter() - t0
        brief_l = tok_b.decode(out_l[0], skip_special_tokens=True)
        # teacher-forced reconstruction accuracy
        d_ids = tok_b(directives, return_tensors="pt", add_special_tokens=False).input_ids.to(model_b.device)
        xr, _, _ = b_inputs(tok_b, model_b, lat, state, "\nThe supervisor's directives are:\n")
        xr = torch.cat([xr, E(d_ids)], 1)
        pred = model_b(inputs_embeds=xr).logits[0, xr.size(1) - d_ids.size(1) - 1:-1].argmax(-1)
        accs.append((pred == d_ids[0]).float().mean().item())
        # --- text path: A decodes directives, B reads them as text
        t0 = time.perf_counter()
        a_text = hfbackend.chat([{"role": "user", "content": plan_prompt(state)}], 0.0, len(d_ids[0]) + 8, model_name=model_a.name_or_path)
        full = hfbackend.render(tok_b, [{"role": "user", "content": brief_prompt(state, a_text)}])
        ids = tok_b(full, return_tensors="pt").to(model_b.device)
        out_t = model_b.generate(**ids, max_new_tokens=gen_tokens, do_sample=False, pad_token_id=tok_b.pad_token_id)
        lat_t += time.perf_counter() - t0
        brief_t = tok_b.decode(out_t[0, ids.input_ids.size(1):], skip_special_tokens=True)
        # does the brief mention the directive titles?
        try:
            titles = [d.get("title", "") for d in json.loads(directives)]
        except Exception:
            titles = []
        for ti in titles:
            key = " ".join(ti.lower().split()[:3])
            n_titles += 1; hits_t += key in brief_t.lower(); hits_l += key in brief_l.lower()
    n = len(data)
    res = {"directive_token_acc_from_latents": sum(accs) / n,
           "brief_mentions_directives_text": hits_t / max(1, n_titles),
           "brief_mentions_directives_latent": hits_l / max(1, n_titles),
           "latency_s_text": lat_t / n, "latency_s_latent": lat_l / n,
           "sample_brief_text": brief_t[:500], "sample_brief_latent": brief_l[:500], "sample_directives": directives[:400]}
    log(json.dumps({k: v for k, v in res.items() if not k.startswith("sample")}, indent=1))
    return res




# ====================================================================================== stage 1: latentcheck

HEADER = re.compile(r"^\s*(#+\s*1\.|##\s*1|\*\*1\.|1\.\s*Learn)", re.I)
def follows_format(t): return bool(HEADER.search(t)) or ("## 1" in t[:200]) or ("Learn" in t[:120] and "{" not in t[:60] and "```" not in t[:20])
THRESH = {"fidelity": 0.70, "decay": 0.90, "format_keep": 0.80, "cue_indep": 0.20, "fix": 0.80}

def run_latentcheck(args):
    rng = random.Random(1); random.seed(1); torch.manual_seed(1)
    tok_a, A = Q.hfbackend.load(agent="FLYWHEEL"); tok_b, B = Q.hfbackend.load(agent="BRIEF")
    tok_a.padding_side = tok_b.padding_side = "left"; dev = A.device; E = Q.embed(B); d_a, d_b = A.config.hidden_size, B.config.hidden_size
    cue = "\nThe supervisor's directives are:\n"
    R = {"sender": A.name_or_path, "receiver": B.name_or_path, "d_sender": d_a, "d_receiver": d_b, "k": args.k, "n": args.n, "hops": args.hops, "probes": {}}
    print(f"latentcheck  sender={A.name_or_path} (d={d_a})  receiver={B.name_or_path} (d={d_b})")

    # ---- data
    print(f"[data] {args.n} plans + text-path outputs ...")
    data = []
    for i in range(args.n):
        s = Q.synth_state(rng)
        plan = Q.hfbackend.chat([{"role": "user", "content": Q.plan_prompt(s)}], 0.0, 300, model_name=A.name_or_path)
        brief = Q.hfbackend.chat([{"role": "user", "content": Q.brief_prompt(s, plan)}], 0.0, 160, model_name=B.name_or_path)
        data.append((s, plan, brief))
    split = int(0.75 * len(data)); tr, te = data[:split], data[split:]
    cache = {id(s): Q.a_hidden(tok_a, A, [Q.plan_prompt(s)]) for s, _, _ in data}
    text_fmt = sum(follows_format(b) for _, _, b in te) / len(te)

    def fidelity(lat, s, plan):
        ids = tok_b(plan, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        x, _, _ = Q.b_inputs(tok_b, B, lat, s, cue); x = torch.cat([x, E(ids)], 1)
        lg = B(inputs_embeds=x).logits.float(); n = x.size(1) - ids.size(1); sl = lg[0, n - 1:-1]
        return F.cross_entropy(sl, ids[0]), (sl.argmax(-1) == ids[0]).float().mean().item()
    def brief_loss(lat, s, brief):
        ids = tok_b(brief, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        x, _, _ = Q.b_inputs(tok_b, B, lat, s, ""); x = torch.cat([x, E(ids)], 1)
        lg = B(inputs_embeds=x).logits.float(); n = x.size(1) - ids.size(1)
        return F.cross_entropy(lg[0, n - 1:-1], ids[0])
    @torch.no_grad()
    def gen(lat, s, suffix=""):
        x, _, _ = Q.b_inputs(tok_b, B, lat, s, suffix)
        o = B.generate(inputs_embeds=x, max_new_tokens=args.gen, do_sample=False, pad_token_id=tok_b.pad_token_id)
        return tok_b.decode(o[0], skip_special_tokens=True)
    @torch.no_grad()
    def gen_text(s, plan, suffix=""):
        full = Q.hfbackend.render(tok_b, [{"role": "user", "content": Q.brief_prompt(s, plan)}]) + suffix
        ids = tok_b(full, return_tensors="pt").to(dev)
        o = B.generate(**ids, max_new_tokens=args.gen, do_sample=False, pad_token_id=tok_b.pad_token_id)
        return tok_b.decode(o[0, ids.input_ids.size(1):], skip_special_tokens=True)
    def relay_hidden(lat):
        bos = E(torch.tensor([[tok_b.bos_token_id or tok_b.eos_token_id]], device=dev))
        with torch.no_grad(): return B(inputs_embeds=torch.cat([bos, lat.to(E.weight.dtype)], 1), output_hidden_states=True).hidden_states[-1].float()
    def train(br, make_lat, lam=0.0):
        opt = torch.optim.AdamW(br.parameters(), lr=1e-3)
        for _ in range(args.steps):
            s, plan, brief = random.choice(tr); lat = make_lat(s)
            loss = fidelity(lat, s, plan)[0] + (lam * brief_loss(lat, s, brief) if lam > 0 else 0)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(br.parameters(), 1.0); opt.step()
        br.eval(); return br
    def eval_fid(make_lat):
        with torch.no_grad(): return sum(fidelity(make_lat(s), s, p)[1] for s, p, _ in te) / len(te)
    def eval_fmt(make_lat, suffix=""):
        with torch.no_grad(): outs = [gen(make_lat(s), s, suffix) for s, _, _ in te]
        return sum(map(follows_format, outs)) / len(outs), outs[0][:160]

    # ---- probe 1: fidelity (one hop)
    t0 = time.time(); br0 = Q.Bridge(d_a, d_b, k=args.k).to(dev); train(br0, lambda s: br0(*cache[id(s)]))
    fid1 = eval_fid(lambda s: br0(*cache[id(s)]))
    R["probes"]["fidelity"] = {"value": fid1, "pass": fid1 >= THRESH["fidelity"], "note": f"one hop, K={args.k}, {args.steps} steps, bridge only"}
    print(f"[fidelity]    {fid1:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- probe 2: decay over hops
    t0 = time.time(); bridges = [br0]; curve = [fid1]
    def chain(s, upto):
        lat = bridges[0](*cache[id(s)])
        for i in range(1, upto): lat = bridges[i](relay_hidden(lat.detach()), None)
        return lat
    for hop in range(2, args.hops + 1):
        br = Q.Bridge(d_b, d_b, k=args.k).to(dev); bridges.append(br)
        for p in [q for b in bridges[:-1] for q in b.parameters()]: p.requires_grad = False
        train(br, lambda s, h=hop: chain(s, h)); curve.append(eval_fid(lambda s, h=hop: chain(s, h)))
    ratio = curve[-1] / max(curve[0], 1e-9)
    R["probes"]["decay"] = {"value": ratio, "curve": curve, "pass": ratio >= THRESH["decay"], "note": f"fidelity at hop {args.hops} / hop 1, no text between hops"}
    print(f"[decay]       hop{args.hops}/hop1 = {ratio:.3f}   curve {[round(c,3) for c in curve]}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- probe 3: format keep + probe 4: cue independence
    t0 = time.time()
    lat_fmt, samp = eval_fmt(lambda s: br0(*cache[id(s)])); lat_fmt_cue, _ = eval_fmt(lambda s: br0(*cache[id(s)]), cue)
    with torch.no_grad(): text_fmt_cue = sum(follows_format(gen_text(s, p, cue)) for s, p, _ in te) / len(te)
    keep = lat_fmt / max(text_fmt, 1e-9)
    R["probes"]["format_keep"] = {"value": keep, "latent": lat_fmt, "text": text_fmt, "pass": keep >= THRESH["format_keep"], "sample_latent": samp, "note": "receiver follows its own output instructions after a latent handoff, relative to the text path"}
    # cue independence: if text+cue stays high while latent collapses, the leak is in the latents (cue innocent)
    cue_delta = text_fmt - text_fmt_cue
    R["probes"]["cue_indep"] = {"value": cue_delta, "text_with_cue": text_fmt_cue, "latent_with_cue": lat_fmt_cue, "pass": cue_delta <= THRESH["cue_indep"],
                                "note": "drop in TEXT-path format-following when the training cue is appended; small = cue innocent, leak (if any) is in the latents"}
    print(f"[format_keep] latent {lat_fmt:.2f} / text {text_fmt:.2f} = {keep:.2f}   | [cue_indep] text+cue {text_fmt_cue:.2f} (delta {cue_delta:+.2f}), latent+cue {lat_fmt_cue:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- probe 5: fix (format-neutral objective)
    if not args.skip_fix:
        t0 = time.time(); brf = Q.Bridge(d_a, d_b, k=args.k).to(dev); train(brf, lambda s: brf(*cache[id(s)]), lam=args.lam)
        fidf = eval_fid(lambda s: brf(*cache[id(s)])); fmtf, sampf = eval_fmt(lambda s: brf(*cache[id(s)]))
        keepf = fmtf / max(text_fmt, 1e-9)
        R["probes"]["fix"] = {"value": keepf, "fidelity": fidf, "fidelity_cost": fid1 - fidf, "format": fmtf, "lambda": args.lam, "pass": keepf >= THRESH["fix"] and (fid1 - fidf) <= 0.05, "sample": sampf,
                              "note": f"mixed objective (fidelity + {args.lam} x receiver-output loss): format relative to text, and fidelity paid"}
        print(f"[fix]         lambda={args.lam}: format {fmtf:.2f} ({keepf:.2f} of text)  fidelity {fidf:.3f} (cost {fid1-fidf:+.3f})  ({time.time()-t0:.0f}s)", flush=True)

    # ---- probe 6: latency
    s, plan, _ = te[0]
    def bench(fn, reps=5):
        fn(); torch.cuda.synchronize() if dev.type == "cuda" else None; t = time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize() if dev.type == "cuda" else None; return (time.perf_counter() - t) / reps * 1000
    with torch.no_grad():
        lat0 = br0(*cache[id(s)])
        R["probes"]["latency"] = {"fast_handoff_ms": bench(lambda: br0(*Q.a_hidden(tok_a, A, [Q.plan_prompt(s)]))), "fast_hop_ms": bench(lambda: bridges[min(1, len(bridges)-1)](relay_hidden(lat0), None)),
                                  "text_decode_ms": bench(lambda: Q.hfbackend.chat([{"role": "user", "content": Q.plan_prompt(s)}], 0.0, 200, model_name=A.name_or_path), 3)}
    L = R["probes"]["latency"]; L["speedup_per_hop"] = L["text_decode_ms"] / max(L["fast_hop_ms"], 1e-9)
    print(f"[latency]     handoff {L['fast_handoff_ms']:.1f} ms | hop {L['fast_hop_ms']:.1f} ms | text {L['text_decode_ms']:.0f} ms | {L['speedup_per_hop']:.0f}x per hop")

    R["verdict"] = "PASS" if all(p.get("pass", True) for p in R["probes"].values()) else "ATTENTION"
    R["label"] = "one seed; synthetic planning task; frozen endpoints; bridges only"
    json.dump(R, open(args.out + ".json", "w"), indent=1); open(args.out + ".html", "w").write(render_card(R))
    print(f"\nverdict: {R['verdict']}   -> {args.out}.json / {args.out}.html"); return R

def render_card(R):
    def row(name, p):
        ok = p.get("pass"); badge = "" if ok is None else (f'<span class="ok">pass</span>' if ok else f'<span class="bad">attention</span>')
        v = p.get("value"); val = f"{v:.3f}" if isinstance(v, (int, float)) else ""
        extra = {k: v for k, v in p.items() if k not in ("value", "pass", "note", "curve") and not str(k).startswith("sample")}
        ex = ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in extra.items())
        curve = ""
        if "curve" in p:
            pts = p["curve"]; w, h = 220, 40
            xs = [i * (w - 10) / max(len(pts) - 1, 1) + 5 for i in range(len(pts))]; ys = [h - 5 - (c * (h - 10)) for c in pts]
            curve = f'<svg width="{w}" height="{h}"><polyline fill="none" stroke="currentColor" stroke-width="2" points="{" ".join(f"{x:.0f},{y:.0f}" for x, y in zip(xs, ys))}"/></svg>'
        samp = p.get("sample_latent") or p.get("sample")
        s = f'<div class="samp">{html.escape(samp)}</div>' if samp else ""
        return f'<tr><td class="n">{name}</td><td class="v">{val}</td><td>{badge}</td><td class="d">{html.escape(p.get("note",""))}<br><small>{html.escape(ex)}</small>{curve}{s}</td></tr>'
    rows = "".join(row(k, v) for k, v in R["probes"].items())
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>latentcheck report</title>
<style>:root{{--bg:#fff;--fg:#111;--mut:#666;--ok:#0a7;--bad:#c40;--line:#e5e5e5}}@media(prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#999;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:900px;margin:auto}}h1{{font-size:22px;margin:0 0 4px}}.sub{{color:var(--mut);margin-bottom:18px}}
table{{width:100%;border-collapse:collapse}}td{{padding:10px 8px;border-top:1px solid var(--line);vertical-align:top}}.n{{font-weight:600;white-space:nowrap}}.v{{font-variant-numeric:tabular-nums;font-size:18px}}
.ok{{color:var(--ok);font-weight:600}}.bad{{color:var(--bad);font-weight:600}}.d{{color:var(--mut)}}.samp{{font-family:ui-monospace,monospace;font-size:12px;background:rgba(127,127,127,.12);padding:6px;border-radius:4px;margin-top:6px;white-space:pre-wrap}}
.verdict{{display:inline-block;padding:4px 10px;border-radius:6px;font-weight:700;background:rgba(127,127,127,.15)}}</style></head><body>
<h1>latentcheck <span class="verdict">{R["verdict"]}</span></h1>
<div class="sub">{html.escape(R["sender"])} (d={R["d_sender"]}) &rarr; {html.escape(R["receiver"])} (d={R["d_receiver"]}) &middot; K={R["k"]} &middot; n={R["n"]} &middot; hops={R["hops"]}<br>{html.escape(R["label"])}</div>
<table>{rows}</table></body></html>'''


# ====================================================================================== incident task (data + prompts)

# ---------------------------------------------------------------- incidents
SERVICES = ["checkout-api", "auth-service", "payments-db", "search-index", "cdn-edge", "notif-worker", "orders-db", "gateway"]
# (alert template, severity, action, known_pattern)
# Templates carry {n} slots filled with random numbers so no two incidents share a string.
# TEST templates are never seen in training (template holdout), so accuracy cannot be template->label lookup.
CRITICAL = [
    ("{s}: database unreachable — connection refused on {n1}", 5, "page", "yes"),
    ("{s}: error rate {n2}% over last 5m (threshold 5%)", 5, "page", "yes"),
    ("{s}: OOM-killed {n3} times in 10m, pods crash-looping", 4, "page", "yes"),
    ("{s}: p99 latency {n4}s (SLO 800ms), queue depth {n5}k and rising", 4, "page", "yes"),
    ("{s}: disk {n6}% full on primary, writes failing", 5, "page", "yes"),
    ("{s}: TLS certificate EXPIRED {n7}m ago — all requests failing", 5, "page", "no"),
    ("{s}: primary lost quorum, {n3} of 5 nodes down, writes rejected", 5, "page", "no"),
    ("{s}: 5xx on {n2}% of requests, upstream timeouts", 5, "page", "yes"),
]
AUTO = [
    ("{s}: disk {n8}% full on replica-2", 2, "auto", "yes"),
    ("{s}: single pod unhealthy, {n9}/{n10} healthy", 1, "auto", "yes"),
    ("{s}: cache hit rate dropped to {n11}% (normal 90%)", 2, "auto", "yes"),
    ("{s}: deploy canary failed health check, auto-rollback available", 3, "auto", "yes"),
    ("{s}: log volume {n12}x baseline, no error spike", 1, "auto", "no"),
    ("{s}: {n13} stale connections in pool, recycling scheduled", 1, "auto", "yes"),
    ("{s}: memory {n14}% on one replica, others nominal", 2, "auto", "yes"),
]
IGNORE = [
    ("{s}: certificate renews in {n15} days (auto-renew enabled)", 0, "ignore", "yes"),
    ("{s}: scheduled backup completed in {n16}m (usual 45m)", 0, "ignore", "yes"),
    ("{s}: CPU {n17}% during nightly batch (expected)", 0, "ignore", "yes"),
    ("{s}: {n18} retried request(s) in last hour", 0, "ignore", "yes"),
    ("{s}: autoscaler cooldown {n19}s remaining", 0, "ignore", "yes"),
    ("{s}: {n18} slow query logged (>1s), within budget", 0, "ignore", "yes"),
]
NUMS = {"n1": (5000, 9999), "n2": (25, 70), "n3": (2, 4), "n4": (5, 20), "n5": (20, 90), "n6": (97, 99), "n7": (3, 120),
        "n8": (82, 91), "n9": (9, 15), "n10": (10, 16), "n11": (60, 78), "n12": (2, 5), "n13": (40, 400), "n14": (80, 92),
        "n15": (20, 60), "n16": (46, 55), "n17": (55, 75), "n18": (1, 3), "n19": (30, 300)}
def _fill(tmpl, rng):
    vals = {k: rng.randint(*v) for k, v in NUMS.items()}
    if "n10" in tmpl: vals["n9"] = vals["n10"] - 1
    return tmpl.format(s=rng.choice(SERVICES), **vals)
def split_templates(rng, hold=2):
    """Hold out `hold` templates per category for test. Returns (train_pool, test_pool)."""
    tr, te = [], []
    for cat in (CRITICAL, AUTO, IGNORE):
        c = cat[:]; rng.shuffle(c); te += c[:hold]; tr += c[hold:]
    return tr, te
NOISE = ["{s}: heartbeat ok", "{s}: config reloaded", "{s}: autoscaler added 1 replica", "{s}: metrics exporter restarted (routine)",
         "{s}: 2 slow queries logged (>1s)", "{s}: cache warmup finished"]
ACTIONS = ["auto", "page", "ignore"]

def make_incident(rng, pool):
    # weight toward page incidents so the frame-leak cells have power
    weights = [3 if a == "page" else (2 if a == "auto" else 1) for _, _, a, _ in pool]
    tmpl, sev, act, known = rng.choices(pool, weights=weights)[0]
    main = _fill(tmpl, rng)
    lines = [rng.choice(NOISE).format(s=rng.choice(SERVICES)) for _ in range(rng.randint(3, 7))]
    lines.insert(rng.randint(0, len(lines)), main)
    storm = "\n".join(f"[{rng.randint(0,23):02d}:{rng.randint(0,59):02d}] {l}" for l in lines)
    return {"storm": storm, "severity": sev, "action": act, "known": known}

NEUTRAL = "You are the on-call Watcher. Read the alert storm and describe what is happening in 2-3 sentences."
PRESSURE = ("You are the on-call Watcher. The team is under SLA pressure and exhausted; leadership has asked to avoid paging "
            "anyone tonight unless absolutely certain. Read the alert storm and describe what is happening in 2-3 sentences.")

def watcher_msgs(storm, frame):
    return [{"role": "system", "content": PRESSURE if frame == "pressure" else NEUTRAL},
            {"role": "user", "content": "ALERTS:\n" + storm}]

# ---------------------------------------------------------------- decider (Jev-style: typed, schema-masked)
PRIMS = {"severity": ("Severity (0-5):", ["0", "1", "2", "3", "4", "5"]),
         "action":   ("Action (auto/page/ignore):", ACTIONS),
         "known":    ("Known pattern (yes/no):", ["yes", "no"])}

# ====================================================================================== stage 2: tree (fan-out / fan-in)

TREE_ROLES = {
    "diagnosis":  "You are the Diagnosis specialist. In one or two sentences, state the single most critical failure and its likely cause.",
    "runbook":    "You are the Runbook specialist. Write exactly 3 numbered remediation steps, nothing else.",
    "escalation": "You are the Escalation specialist. In one sentence say who should be paged (or that no page is needed) and the tone to use.",
}
COMMANDER = ("You are the Incident Commander. Using the specialists' input, write a 4-line incident runbook with exactly these labels: "
             "Impact:, Action:, Owner:, Follow-up:")
RUNBOOK_RE = re.compile(r"Impact\s*:.*Action\s*:", re.S | re.I)

def pick_lambda(path="out/night3_results.json", default=1.0):
    try:
        m = json.load(open(path))["mixed"]; base = m["lambda=0"]["fidelity"]
        ok = [(v["format_rate"], -abs(float(k.split("=")[1])), float(k.split("=")[1])) for k, v in m.items() if v["fidelity"] >= base - 0.02]
        return max(ok)[2] if ok else default
    except Exception:
        return default

def run_tree(args):
    lam = args.lam if args.lam is not None else pick_lambda()
    rng = random.Random(11); random.seed(11); torch.manual_seed(11)
    tok_w, W = Q.hfbackend.load(agent="FLYWHEEL"); tok_d, D = Q.hfbackend.load(agent="BRIEF")
    tok_w.padding_side = tok_d.padding_side = "left"; dev = W.device; E = Q.embed(D)
    d_w, d_d, K = W.config.hidden_size, D.config.hidden_size, args.k
    print(f"watcher={W.name_or_path}  specialists/commander={D.name_or_path}  K={K}  lambda={lam}")

    # ---------------------------------------------------------------- data + TEXT TREE (targets)
    tr_pool, te_pool = IR.split_templates(rng); n_tr = int(0.6 * args.n)
    train = [IR.make_incident(rng, tr_pool) for _ in range(n_tr)]; test = [IR.make_incident(rng, te_pool) for _ in range(args.n - n_tr)]
    incidents = train + test
    print(f"{len(train)} train / {len(test)} test (held-out templates). Running the TEXT tree to produce targets ...")
    def chat_d(system, user, max_new): return Q.hfbackend.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], 0.0, max_new, model_name=D.name_or_path)
    t_text_tree = []
    for i, inc in enumerate(incidents):
        t0 = time.perf_counter()
        inc["summary"] = Q.hfbackend.chat(IR.watcher_msgs(inc["storm"], "neutral"), 0.0, args.sum_tokens, model_name=W.name_or_path)
        inc["spec"] = {r: chat_d(TREE_ROLES[r], "Watcher summary:\n" + inc["summary"], 60) for r in TREE_ROLES}
        inc["cmd_text"] = chat_d(COMMANDER, "\n".join(f"{r.capitalize()}: {inc['spec'][r]}" for r in TREE_ROLES), args.gen)
        t_text_tree.append(time.perf_counter() - t0)
        if i % 40 == 0: print(f"  {i}/{args.n}  commander: {inc['cmd_text'][:70]!r}", flush=True)
    hid = {}
    for inc in incidents:
        enc = tok_w(Q.hfbackend.render(tok_w, IR.watcher_msgs(inc["storm"], "neutral")), return_tensors="pt", truncation=True, max_length=1024).to(dev)
        with torch.no_grad(): hid[id(inc)] = W(**enc, output_hidden_states=True).hidden_states[-1].float()

    # ---------------------------------------------------------------- embedding helpers
    def around(system, user_with_slot, lat, suffix=""):
        full = Q.hfbackend.render(tok_d, [{"role": "system", "content": system}, {"role": "user", "content": user_with_slot}]) + suffix
        pre, post = full.split(SLOT)
        pre_ids = tok_d(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        post_ids = tok_d(post, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        return torch.cat([E(pre_ids), lat.to(E.weight.dtype), E(post_ids)], 1)
    def tf_loss(x, target_text):
        ids = tok_d(target_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=120).input_ids.to(dev)
        xx = torch.cat([x.to(E.weight.dtype), E(ids)], 1); lg = D(inputs_embeds=xx).logits.float(); n = xx.size(1) - ids.size(1); sl = lg[0, n - 1:-1]
        return F.cross_entropy(sl, ids[0]), (sl.argmax(-1) == ids[0]).float().mean().item()
    ans_ids = {p: [[tok_d(" " + o, add_special_tokens=False).input_ids[0], tok_d(o, add_special_tokens=False).input_ids[0]] for o in opts] for p, (q, opts) in IR.PRIMS.items()}
    def decide(x, prim):
        xx = torch.cat([x.to(E.weight.dtype), E(tok_d("\n" + IR.PRIMS[prim][0], return_tensors="pt", add_special_tokens=False).input_ids.to(dev))], 1)
        lg = D(inputs_embeds=xx).logits[0, -1].float()
        return torch.stack([torch.logsumexp(lg[c], 0) for c in ans_ids[prim]])
    def label(inc, prim): return {"severity": inc["severity"], "action": IR.ACTIONS.index(inc["action"]), "known": 0 if inc["known"] == "yes" else 1}[prim]

    # ---------------------------------------------------------------- latent tree modules
    br_in = nn.ModuleDict({r: Q.Bridge(d_w, d_d, k=K) for r in TREE_ROLES}).to(dev)        # fan-out (broadcast readers)
    br_out = nn.ModuleDict({r: Q.Bridge(d_d, d_d, k=K) for r in TREE_ROLES}).to(dev)       # specialist state -> K
    merge = nn.Linear(3 * d_d, d_d).to(dev)   # naive fan-in, PER SLOT: [A_j | B_j | C_j] (3d) -> d  (12.6M params, not 3.2B)
    cue = "\nThe watcher's summary was:\n"
    def spec_x(r, lat): return around(TREE_ROLES[r], "Watcher state:\n" + SLOT, lat)
    def spec_hidden(r, lat):
        with torch.no_grad(): return D(inputs_embeds=spec_x(r, lat), output_hidden_states=True).hidden_states[-1].float()
    def merged(inc, drop=None):
        outs = []
        for r in TREE_ROLES:
            lat = br_in[r](hid[id(inc)], None); o = br_out[r](spec_hidden(r, lat), None)
            outs.append(torch.zeros_like(o) if r == drop else o)
        return merge(torch.cat(outs, -1))                       # (1, K, 3d) -> (1, K, d)
    def cmd_x(lat): return around(COMMANDER, "Specialist states:\n" + SLOT, lat)

    # ---- stage 1: branch bridges, mixed objective
    print(f"\nstage 1: branch bridges (mixed objective, lambda={lam}) ...")
    for r in TREE_ROLES:
        opt = torch.optim.AdamW(br_in[r].parameters(), lr=1e-3); t0 = time.time()
        for step in range(args.steps):
            inc = random.choice(train); lat = br_in[r](hid[id(inc)], None)
            loss, _ = tf_loss(spec_x(r, lat), inc["spec"][r])                                   # own role output
            if lam > 0: loss = loss + lam * tf_loss(around(TREE_ROLES[r], "Watcher state:\n" + SLOT, lat, cue), inc["summary"])[0]
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(br_in[r].parameters(), 1.0); opt.step()
        br_in[r].eval(); print(f"  {r:11s} bridge trained  loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    for p in br_in.parameters(): p.requires_grad = False
    # ---- stage 2: out-bridges + merge + commander objective (runbook + Jev decisions)
    print("stage 2: fan-in (out-bridges + merge) on commander objective ...")
    params = list(br_out.parameters()) + list(merge.parameters()); opt = torch.optim.AdamW(params, lr=1e-3); t0 = time.time()
    for step in range(args.steps):
        inc = random.choice(train); x = cmd_x(merged(inc))
        loss, _ = tf_loss(x, inc["cmd_text"])
        for prim in IR.PRIMS: loss = loss + F.cross_entropy(decide(x, prim).unsqueeze(0), torch.tensor([label(inc, prim)], device=dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 100 == 0: print(f"  step {step} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    br_out.eval()

    # ---------------------------------------------------------------- eval
    res = {"models": {"watcher": W.name_or_path, "executor": D.name_or_path}, "k": K, "lambda": lam, "n_test": len(test), "design": {"fan_out": "broadcast", "merge": "per-slot concat [A_j|B_j|C_j] (3d) -> linear -> d", "branch_objective": "own-role CE + lambda*summary CE", "template_holdout": True, "seed": 11}}
    def ece(probs, labels, bins=10):
        conf, pred = probs.max(-1); acc = (pred == labels).float(); e = 0.0
        for i in range(bins):
            m = (conf > i / bins) & (conf <= (i + 1) / bins)
            if m.any(): e += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
        return e
    def eval_tree(ctx_fn, name):
        P = {p: [] for p in IR.PRIMS}; Y = {p: [] for p in IR.PRIMS}; rb_acc = []; rb_ce = []; fmt = 0; samples = []
        with torch.no_grad():
            for inc in test:
                x = ctx_fn(inc)
                for prim in IR.PRIMS: P[prim].append(decide(x, prim).softmax(-1)); Y[prim].append(label(inc, prim))
                ce, acc = tf_loss(x, inc["cmd_text"]); rb_ce.append(ce.item()); rb_acc.append(acc)
                if len(samples) < 6:
                    out = D.generate(inputs_embeds=x.to(E.weight.dtype), max_new_tokens=args.gen, do_sample=False, pad_token_id=tok_d.pad_token_id)
                    txt = tok_d.decode(out[0], skip_special_tokens=True); samples.append(txt[:200]); fmt += bool(RUNBOOK_RE.search(txt))
        out = {p: {"acc": (torch.stack(P[p]).argmax(-1) == torch.tensor(Y[p], device=dev)).float().mean().item(), "ece": ece(torch.stack(P[p]), torch.tensor(Y[p], device=dev))} for p in IR.PRIMS}
        out["runbook_token_acc"] = sum(rb_acc) / len(rb_acc); out["runbook_ce"] = sum(rb_ce) / len(rb_ce); out["runbook_format_rate_6"] = fmt / 6; out["samples"] = samples[:2]
        print(f"  {name:22s}: " + "  ".join(f"{p} {out[p]['acc']:.3f}/ECE {out[p]['ece']:.3f}" for p in IR.PRIMS) + f"  | runbook tok-acc {out['runbook_token_acc']:.3f} CE {out['runbook_ce']:.2f} fmt {out['runbook_format_rate_6']:.2f}")
        return out
    print("\n=== RESULTS ===")
    def text_ctx(inc):
        full = Q.hfbackend.render(tok_d, [{"role": "system", "content": COMMANDER}, {"role": "user", "content": "\n".join(f"{r.capitalize()}: {inc['spec'][r]}" for r in TREE_ROLES)}])
        return E(tok_d(full, return_tensors="pt", add_special_tokens=False).input_ids.to(dev))
    res["text_tree"] = eval_tree(text_ctx, "TEXT tree")
    res["latent_tree"] = eval_tree(lambda inc: cmd_x(merged(inc)), "LATENT tree")
    print("  leave-one-branch-out (latent tree):")
    res["ablation"] = {}
    for r in TREE_ROLES:
        o = eval_tree(lambda inc, r=r: cmd_x(merged(inc, drop=r)), f"  drop {r}")
        res["ablation"][r] = {"d_action_acc": o["action"]["acc"] - res["latent_tree"]["action"]["acc"], "d_severity_acc": o["severity"]["acc"] - res["latent_tree"]["severity"]["acc"], "d_runbook_ce": o["runbook_ce"] - res["latent_tree"]["runbook_ce"]}
    # latency: full tree, one incident
    inc = test[0]
    def bench(fn, reps=5):
        fn(); torch.cuda.synchronize() if dev.type == "cuda" else None; t = time.perf_counter()
        for _ in range(reps): fn()
        torch.cuda.synchronize() if dev.type == "cuda" else None; return (time.perf_counter() - t) / reps * 1000
    def text_tree_once():
        s = Q.hfbackend.chat(IR.watcher_msgs(inc["storm"], "neutral"), 0.0, args.sum_tokens, model_name=W.name_or_path)
        sp = {r: chat_d(TREE_ROLES[r], "Watcher summary:\n" + s, 60) for r in TREE_ROLES}
        x = text_ctx({"spec": sp}); return [decide(x, p) for p in IR.PRIMS]
    def latent_tree_once():
        enc = tok_w(Q.hfbackend.render(tok_w, IR.watcher_msgs(inc["storm"], "neutral")), return_tensors="pt").to(dev)
        h = W(**enc, output_hidden_states=True).hidden_states[-1].float(); hid[id(inc)] = h
        x = cmd_x(merged(inc)); return [decide(x, p) for p in IR.PRIMS]
    with torch.no_grad():
        res["latency_ms"] = {"text_tree_to_decision": bench(text_tree_once, 3), "latent_tree_to_decision": bench(latent_tree_once)}
    res["latency_ms"]["speedup"] = res["latency_ms"]["text_tree_to_decision"] / res["latency_ms"]["latent_tree_to_decision"]
    print(f"  latency to Commander decision: text tree {res['latency_ms']['text_tree_to_decision']:.0f} ms | latent tree {res['latency_ms']['latent_tree_to_decision']:.1f} ms | {res['latency_ms']['speedup']:.0f}x")
    # ---- voice-turn budget: does the whole 3-specialist tree fit inside one conversational turn?
    # A voice assistant has ~300-500 ms before the pause sounds broken. Per-incident time to Commander decision, each lane.
    B = args.voice_budget_ms; nv = min(20, len(test)); v = {"budget_ms": B, "n": nv}
    with torch.no_grad():
        def text_once_v():
            s = Q.hfbackend.chat(IR.watcher_msgs(inc["storm"], "neutral"), 0.0, args.sum_tokens, model_name=W.name_or_path)
            sp = {r: chat_d(TREE_ROLES[r], "Watcher summary:\n" + s, 60) for r in TREE_ROLES}
            x = text_ctx({"spec": sp}); return [decide(x, p) for p in IR.PRIMS]
        def latent_once_v():
            enc = tok_w(Q.hfbackend.render(tok_w, IR.watcher_msgs(inc["storm"], "neutral")), return_tensors="pt").to(dev)
            hid[id(inc)] = W(**enc, output_hidden_states=True).hidden_states[-1].float()
            x = cmd_x(merged(inc)); return [decide(x, p) for p in IR.PRIMS]
        tl = []; ll = []
        for inc in test[:nv]:
            torch.cuda.synchronize() if dev.type == "cuda" else None; t = time.perf_counter(); text_once_v(); torch.cuda.synchronize() if dev.type == "cuda" else None; tl.append((time.perf_counter() - t) * 1000)
            torch.cuda.synchronize() if dev.type == "cuda" else None; t = time.perf_counter(); latent_once_v(); torch.cuda.synchronize() if dev.type == "cuda" else None; ll.append((time.perf_counter() - t) * 1000)
    for name, arr in (("text", tl), ("latent", ll)):
        s = sorted(arr); v[name] = {"median_ms": s[len(s) // 2], "p90_ms": s[int(0.9 * (len(s) - 1))], "in_budget_rate": sum(a <= B for a in arr) / len(arr)}
    v["survives"] = {"accuracy": {p: res["latent_tree"][p]["acc"] - res["text_tree"][p]["acc"] for p in IR.PRIMS}, "format": res["latent_tree"]["runbook_format_rate_6"]}
    res["voice_turn"] = v
    print(f"  voice-turn budget {B:.0f} ms (n={nv}): text tree in-budget {v['text']['in_budget_rate']:.2f} (median {v['text']['median_ms']:.0f} ms) | latent tree in-budget {v['latent']['in_budget_rate']:.2f} (median {v['latent']['median_ms']:.0f} ms, p90 {v['latent']['p90_ms']:.0f} ms)")
    print("  survives under budget: " + "  ".join(f"{p} {v['survives']['accuracy'][p]:+.3f} vs text" for p in IR.PRIMS) + f"  | runbook format {v['survives']['format']:.2f}")
    for r, a in res["ablation"].items(): print(f"  drop {r:11s}: d(action acc) {a['d_action_acc']:+.3f}  d(severity acc) {a['d_severity_acc']:+.3f}  d(runbook CE) {a['d_runbook_ce']:+.2f}")
    os.makedirs("out", exist_ok=True); json.dump(res, open("out/tree_results.json", "w"), indent=1)
    print("saved out/tree_results.json"); return res


# ====================================================================================== auction environment

HOURS = list(range(24)); REGIONS = list(range(1, 36)); EXCH = [1, 2, 3]
SLOTS = [(300, 250), (728, 90), (160, 600), (336, 280), (950, 90), (320, 50)]
DOMAINS = [f"site{i}" for i in range(1, 41)]
TAGS = [10006, 10024, 10031, 10048, 10052, 10057, 10059, 10063, 10067, 10074, 10075, 10076, 10077, 10079, 10083, 10093, 10102, 10110, 10111, 10684, 11092, 11278, 11379, 11423, 11512, 11576, 11632, 11680, 11724, 11944, 13042, 13403, 13496, 13678, 13776, 13800, 13866, 13874, 14273, 16593, 16617, 16661, 16706]
VALUE_PER_CLICK = 2000.0   # CPM-style units, like iPinYou prices (1000 = $1 per impression-equivalent)

def _sig(x): return 1 / (1 + math.exp(-x))

class World:
    """Fixed hidden weights so the three slices have the pre-set informativeness."""
    def __init__(self, seed=0):
        r = random.Random(seed)
        self.w_tag = {t: r.gauss(0, 0.25) for t in TAGS}; self.w_region = {g: r.gauss(0, 0.15) for g in REGIONS}; self.w_gender = {"m": 0.05, "f": -0.05, "null": 0.0}
        self.w_slot = {s: r.gauss(0, 0.35) for s in SLOTS}; self.w_domain = {d: r.gauss(0, 0.3) for d in DOMAINS}
        self.w_hour = {h: 0.25 * math.sin((h - 6) / 24 * 2 * math.pi) for h in HOURS}; self.w_exch = {1: 0.1, 2: -0.1, 3: 0.0}; self.w_vis = {0: -0.3, 1: 0.1, 2: 0.3}
        self.base = -5.2  # ~0.6% mean CTR
    def ctr(self, a):
        u = sum(self.w_tag[t] for t in a["user"]["tags"]) + self.w_region[a["user"]["region"]] + self.w_gender[a["user"]["gender"]]
        c = self.w_slot[a["context"]["slot"]] + self.w_domain[a["context"]["domain"]] + self.w_hour[a["context"]["hour"]] + self.w_exch[a["context"]["exchange"]] + self.w_vis[a["context"]["visibility"]]
        h = 1.6 * a["history"]["past_ctr_z"] - 0.25 * math.log1p(a["history"]["freq_today"])
        return _sig(self.base + 0.89 * u + 1.15 * c + 0.61 * h), {"user": 0.89 * u, "context": 1.15 * c, "history": 0.61 * h}

def make_auction(rng, world, n_competitors=3, comp_noise=0.35):
    a = {"user": {"tags": sorted(rng.sample(TAGS, rng.randint(2, 7))), "region": rng.choice(REGIONS), "gender": rng.choice(["m", "f", "null"])},
         "context": {"slot": rng.choice(SLOTS), "domain": rng.choice(DOMAINS), "hour": rng.choice(HOURS), "exchange": rng.choice(EXCH), "visibility": rng.choice([0, 1, 2]), "floor": rng.choice([0, 0, 5, 20, 50])},
         "history": {"past_ctr_z": rng.gauss(0, 1), "freq_today": rng.randint(0, 12), "last_seen_min": rng.randint(1, 1440)}}
    ctr, parts = world.ctr(a); value = ctr * VALUE_PER_CLICK
    comps = [value * math.exp(rng.gauss(-0.30, comp_noise)) for _ in range(n_competitors)]   # informed but noisy field; max of 3 lands near true value
    a["hidden"] = {"ctr": ctr, "value": value, "logit_parts": parts, "field": sorted(comps, reverse=True), "clearing": max(max(comps), a["context"]["floor"])}
    return a

def settle(a, bid):
    """Second price. Returns (won, paid, surplus)."""
    if bid is None or bid < a["context"]["floor"]: return False, 0.0, 0.0
    top = a["hidden"]["field"][0]
    if bid > top: return True, top, a["hidden"]["value"] - top
    return False, 0.0, 0.0

# ---------------------------------------------------------------- what each specialist is allowed to see (DISJOINT)
def slice_text(a, which):
    if which == "user": return f"User segment: tags {','.join(map(str, a['user']['tags']))}; region {a['user']['region']}; gender {a['user']['gender']}."
    if which == "context":
        c = a["context"]; return f"Context: slot {c['slot'][0]}x{c['slot'][1]}, visibility {c['visibility']}, domain {c['domain']}, hour {c['hour']:02d}, exchange {c['exchange']}, floor {c['floor']}."
    if which == "history":
        h = a["history"]; return f"History: this user's past CTR is {h['past_ctr_z']:+.2f} std from average; seen {h['freq_today']} times today; last seen {h['last_seen_min']} min ago."
    raise ValueError(which)

# ---------------------------------------------------------------- reference bidders (no LLM) to validate the env
def slice_estimate(a, world, which):
    """Value estimate from ONE slice only: that slice's logit contribution + base (others at their mean 0)."""
    return _sig(world.base + a["hidden"]["logit_parts"][which]) * VALUE_PER_CLICK

def run_reference(n=20000, seed=0):
    rng = random.Random(seed); world = World(seed)
    aucs = [make_auction(rng, world) for _ in range(n)]
    out = {}
    for name, bidder in (("oracle (true value)", lambda a: a["hidden"]["value"]),
                         ("all-slices (sum of 3)", lambda a: _sig(world.base + sum(a["hidden"]["logit_parts"].values())) * VALUE_PER_CLICK),
                         ("history only", lambda a: slice_estimate(a, world, "history")), ("context only", lambda a: slice_estimate(a, world, "context")),
                         ("user only", lambda a: slice_estimate(a, world, "user")), ("pass always", lambda a: None),
                         ("random (value-blind)", lambda a: rng.uniform(0, 40))):
        S = W = 0.0
        for a in aucs:
            won, paid, s = settle(a, bidder(a)); S += s; W += won
        out[name] = {"surplus_per_1k": S / n * 1000, "win_rate": W / n}
    return out, aucs

# ====================================================================================== stage 3: auction (deadline demo)

SLICES = ["user", "context", "history"]
BINS = [0, 4, 8, 16, 32, 64, 128, 256, float("inf")]
def bin_of(v): return next(i for i in range(8) if v < BINS[i + 1])
def center(i): return 2.0 if i == 0 else (300.0 if i == 7 else math.sqrt(BINS[i] * BINS[i + 1]))
AUC_ROLE = {"user": "You are the Audience specialist. From the user segment, say in one sentence how likely this user is to click, and why.",
        "context": "You are the Placement specialist. From the ad slot and page context, say in one sentence how valuable this placement is, and why.",
        "history": "You are the History specialist. From this user's history, say in one sentence how likely a click is now, and why."}
CMD = "You are the Commander bidder in a second-price auction. Using the specialists' input, estimate the impression's value and decide whether to bid."
AUC_PRIMS = {"bucket": ("Value bucket (0-7):", [str(i) for i in range(8)]), "action": ("Decision (bid/pass):", ["bid", "pass"])}

def run_auction(args):
    rng = random.Random(args.seed); random.seed(args.seed); torch.manual_seed(args.seed)
    tok_s, S = Q.hfbackend.load(agent="FLYWHEEL"); tok_d, D = Q.hfbackend.load(agent="BRIEF")
    tok_s.padding_side = tok_d.padding_side = "left"; dev = S.device; E = Q.embed(D); d_s, d_d, K = S.config.hidden_size, D.config.hidden_size, args.k
    print(f"specialists={S.name_or_path} (d={d_s})  commander={D.name_or_path} (d={d_d})  K={K}  deadline={args.deadline_ms}ms control={args.control_ms}ms")

    world = AE.World(args.seed); aucs = [AE.make_auction(rng, world) for _ in range(args.n)]
    n_tr = int(0.6 * args.n); train, test = aucs[:n_tr], aucs[n_tr:]
    for a in aucs: a["y_bucket"] = bin_of(a["hidden"]["value"]); a["y_logv"] = math.log(max(a["hidden"]["value"], 0.5)); a["y_action"] = 0 if a["hidden"]["value"] >= a["context"]["floor"] else 1
    LOGV_MEAN = sum(a["y_logv"] for a in train) / len(train); LOGV_STD = (sum((a["y_logv"] - LOGV_MEAN) ** 2 for a in train) / len(train)) ** 0.5
    print(f"{len(train)} train / {len(test)} test auctions; mean value {sum(a['hidden']['value'] for a in test)/len(test):.1f}")

    # ---------------------------------------------------------------- specialist senders: memos (text lane) and hidden states (latent lane)
    def spec_msgs(a, w): return [{"role": "system", "content": AUC_ROLE[w]}, {"role": "user", "content": AE.slice_text(a, w)}]
    print("specialists writing memos (text lane; each timed) ...")
    for i, a in enumerate(aucs):
        a["memo"], a["memo_ms"] = {}, {}
        for w in SLICES:
            t0 = time.perf_counter(); a["memo"][w] = Q.hfbackend.chat(spec_msgs(a, w), 0.0, args.memo_tokens, model_name=S.name_or_path); a["memo_ms"][w] = (time.perf_counter() - t0) * 1000
        if i % 80 == 0: print(f"  {i}/{args.n}  e.g. {a['memo']['history'][:80]!r}", flush=True)
    print("caching specialist hidden states (latent lane) ...")
    hid = {}
    for a in aucs:
        hid[id(a)] = {}
        for w in SLICES:
            enc = tok_s(Q.hfbackend.render(tok_s, spec_msgs(a, w)), return_tensors="pt").to(dev)
            with torch.no_grad(): hid[id(a)][w] = S(**enc, output_hidden_states=True).hidden_states[-1].float()

    # ---------------------------------------------------------------- commander
    # Jev-style TYPED HEAD: a trained linear readout on the frozen Commander's last hidden state, one head per primitive.
    # (The schema-masked next-token variant collapsed to the majority class in incident_race; a decision model is a head on a backbone.)
    def new_heads():
        return nn.ModuleDict({"value": nn.Linear(d_d, 1), "action": nn.Linear(d_d, 2)}).to(dev)   # value: regress log(value); action: bid/pass
    heads = new_heads()
    def around(user_with_slot, lat):
        full = Q.hfbackend.render(tok_d, [{"role": "system", "content": CMD}, {"role": "user", "content": user_with_slot}])
        pre, post = full.split(SLOT)
        return torch.cat([E(tok_d(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)), lat.to(E.weight.dtype), E(tok_d(post, return_tensors="pt", add_special_tokens=False).input_ids.to(dev))], 1)
    def last_hidden(x):
        """Mean-pooled last-layer hidden state over the whole Commander input (more signal than one token)."""
        xx = torch.cat([x.to(E.weight.dtype), E(tok_d("\nDecide.", return_tensors="pt", add_special_tokens=False).input_ids.to(dev))], 1)
        return D(inputs_embeds=xx, output_hidden_states=True).hidden_states[-1][0].float().mean(0)
    def decide(x, prim, h=None):
        if h is None: h = last_hidden(x)
        return heads[prim](h)
    soft = nn.Parameter(torch.randn(K, d_d, device=dev) * 0.02)
    br_in = nn.ModuleDict({w: Q.Bridge(d_s, d_d, k=K) for w in SLICES}).to(dev); merge = nn.Linear(3 * d_d, d_d).to(dev)
    def ctx_text(a, which=SLICES):
        txt = "\n".join(f"{w.capitalize()} specialist: {a['memo'][w]}" for w in which)
        return around("Specialist memos:\n" + txt + "\n" + SLOT, soft.unsqueeze(0))
    def ctx_blind(a): return around("Context only:\n" + AE.slice_text(a, "context") + "\n" + SLOT, soft.unsqueeze(0))
    def ctx_latent(a, drop=None):
        outs = [torch.zeros(1, K, d_d, device=dev) if w == drop else br_in[w](hid[id(a)][w], None) for w in SLICES]
        return around("Specialist states:\n" + SLOT, merge(torch.cat(outs, -1)))
    def loss_on(x, a):
        h = last_hidden(x)
        z = torch.tensor([(a["y_logv"] - LOGV_MEAN) / LOGV_STD], device=dev)
        return F.mse_loss(decide(x, "value", h), z) + 0.5 * F.cross_entropy(decide(x, "action", h).unsqueeze(0), torch.tensor([a["y_action"]], device=dev))
    def pred_value(h): return math.exp(decide(None, "value", h).item() * LOGV_STD + LOGV_MEAN)
    head_sets = {}
    def fit(params, ctx, name, lr):
        nonlocal heads
        heads = new_heads(); head_sets[name] = heads
        params = list(params) + list(heads.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05); t0 = time.time()
        for step in range(args.steps):
            a = random.choice(train); loss = loss_on(ctx(a), a)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % 100 == 0: print(f"  [{name}] step {step} loss {loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    print("training text soft prompt ..."); fit([soft], ctx_text, "text", 3e-3)
    print("training latent bridges + merge ..."); fit(list(br_in.parameters()) + list(merge.parameters()), ctx_latent, "latent", 1e-3)
    br_in.eval()

    # ---------------------------------------------------------------- eval with per-auction billing
    def ece(P, Y, bins=10):
        conf, pred = P.max(-1); acc = (pred == Y).float(); e = 0.0
        for i in range(bins):
            m = (conf > i / bins) & (conf <= (i + 1) / bins)
            if m.any(): e += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
        return e
    def sync():
        if dev.type == "cuda": torch.cuda.synchronize()
    @torch.no_grad()
    def run_lane(name, ctx_fn, coord_ms_fn, head_set="latent"):
        nonlocal heads
        heads = head_sets[head_set]; rows = []; P = []; Y = []
        for a in test:
            sync(); t0 = time.perf_counter(); x = ctx_fn(a); h = last_hidden(x); v = pred_value(h); pa = decide(x, "action", h).softmax(-1); sync()
            ms = (time.perf_counter() - t0) * 1000 + coord_ms_fn(a)
            act = pa.argmax().item(); bid = v if act == 0 else None
            P.append(pa); Y.append(a["y_action"])
            rows.append({"ms": ms, "bid": bid, "bucket_ok": bin_of(v) == a["y_bucket"], "log_err": abs(math.log(max(v, 0.5)) - a["y_logv"]), "won_paid_surplus": AE.settle(a, bid)})
        out = {}
        for dl in (args.deadline_ms, args.control_ms, float("inf")):
            S_ = W_ = 0.0; miss = 0
            for r in rows:
                if r["ms"] > dl: miss += 1; continue          # timeout => PASS
                won, paid, s = r["won_paid_surplus"]; S_ += s; W_ += won
            out[f"deadline_{'inf' if dl == float('inf') else int(dl)}ms"] = {"surplus_per_1k": S_ / len(test) * 1000, "win_rate": W_ / len(test), "miss_rate": miss / len(test)}
        out["bucket_acc"] = sum(r["bucket_ok"] for r in rows) / len(rows); out["value_log_mae"] = sum(r["log_err"] for r in rows) / len(rows); out["ece_action"] = ece(torch.stack(P), torch.tensor(Y, device=dev))
        ms = sorted(r["ms"] for r in rows); out["ms_median"] = ms[len(ms) // 2]; out["ms_p90"] = ms[int(0.9 * len(ms))]
        d = out[f"deadline_{int(args.deadline_ms)}ms"]; c = out[f"deadline_{int(args.control_ms)}ms"]
        print(f"  {name:28s} @{int(args.deadline_ms)}ms: surplus/1k {d['surplus_per_1k']:8.1f} win {d['win_rate']:.2f} miss {d['miss_rate']:.2f} | @{int(args.control_ms)}ms: surplus/1k {c['surplus_per_1k']:8.1f} miss {c['miss_rate']:.2f} | bucket acc {out['bucket_acc']:.2f} logMAE {out['value_log_mae']:.2f} ECE {out['ece_action']:.3f} | median {out['ms_median']:.0f} ms p90 {out['ms_p90']:.0f} ms")
        return out
    res = {"models": {"specialists": S.name_or_path, "commander": D.name_or_path}, "k": K, "n_test": len(test), "deadline_ms": args.deadline_ms, "control_ms": args.control_ms,
           "design": {"env": "auction_env.py (second price, 3 noisy competitors, slices 55/30/15 by logit variance, disjoint by construction)", "timeout_policy": "pass",
                      "text_lane_billing": "measured memo generation (3 memos) + commander pass, per auction", "latent_lane_billing": "measured specialist forward x3 + bridges + merge + commander pass, per auction",
                      "fair_budget": f"text: K={K} soft prompt + linear typed heads; latent: 3 bridges + merge + linear typed heads; same {args.steps} steps", "decider": "linear typed heads on MEAN-POOLED frozen Commander hidden state: value = regression on standardized log(value) (MSE), action = bid/pass (CE); weight decay 0.05; per lane", "seed": args.seed}, "lanes": {}}
    # reference bidders (no LLM, no deadline)
    ref = {}
    MEANV = math.exp(LOGV_MEAN)
    for nm, bidder in (("oracle", lambda a: a["hidden"]["value"]), ("history_only", lambda a: AE.slice_estimate(a, world, "history")), ("constant_prior", lambda a: MEANV), ("pass", lambda a: None)):
        S_ = sum(AE.settle(a, bidder(a))[2] for a in test); ref[nm] = S_ / len(test) * 1000
    res["reference_surplus_per_1k"] = ref; print("reference (no LLM):", {k: round(v, 1) for k, v in ref.items()})
    print("\n=== LANES ===")
    res["lanes"]["blind"] = run_lane("BLIND (context slice only)", ctx_blind, lambda a: 0.0, head_set="text")
    res["lanes"]["text"] = run_lane("TEXT (3 memos)", ctx_text, lambda a: sum(a["memo_ms"].values()), head_set="text")
    # latent coordination cost: re-run the 3 specialist forwards live (batched would be faster; measured honestly as 3 passes)
    def latent_coord_ms(a):
        sync(); t0 = time.perf_counter()
        for w in SLICES:
            enc = tok_s(Q.hfbackend.render(tok_s, spec_msgs(a, w)), return_tensors="pt").to(dev); S(**enc, output_hidden_states=True)
        sync(); return (time.perf_counter() - t0) * 1000
    res["lanes"]["latent"] = run_lane("LATENT (3 bridges -> merge)", ctx_latent, latent_coord_ms)
    print("  leave-one-branch-out (latent, no deadline):")
    base = res["lanes"]["latent"]["deadline_infms"]["surplus_per_1k"]; res["ablation"] = {}
    for w in SLICES:
        o = run_lane(f"    drop {w}", lambda a, w=w: ctx_latent(a, drop=w), lambda a: 0.0)
        res["ablation"][w] = {"d_surplus_per_1k": o["deadline_infms"]["surplus_per_1k"] - base, "d_bucket_acc": o["bucket_acc"] - res["lanes"]["latent"]["bucket_acc"]}
    for w, d in res["ablation"].items(): print(f"  drop {w:8s}: d(surplus/1k) {d['d_surplus_per_1k']:+8.1f}   d(bucket acc) {d['d_bucket_acc']:+.3f}")
    print("  ground truth informativeness: history > context > user")
    os.makedirs("out", exist_ok=True); json.dump(res, open("out/auction_results.json", "w"), indent=1); print("saved out/auction_results.json"); return res



# ====================================================================================== driver
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sender", default=os.environ.get("SENDER", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--receiver", default=os.environ.get("RECEIVER", "HuggingFaceTB/SmolLM2-1.7B-Instruct"))
    ap.add_argument("--stages", default="latentcheck,tree,auction")
    ap.add_argument("--quick", action="store_true", help="small config for a smoke run")
    ap.add_argument("--n", type=int, default=None); ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--hops", type=int, default=8)
    ap.add_argument("--deadline-ms", type=float, default=100.0); ap.add_argument("--control-ms", type=float, default=3000.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--voice-budget-ms", type=float, default=400.0, help="one conversational turn; the tree must reach a decision inside it")
    a = ap.parse_args()
    os.environ["FLYWHEEL_HF_MODEL"] = a.sender; os.environ["BRIEF_HF_MODEL"] = a.receiver
    os.makedirs("out", exist_ok=True)
    q = a.quick
    cfg = {"latentcheck": dict(n=a.n or (16 if q else 48), steps=a.steps or (60 if q else 300), hops=a.hops if not q else 4, k=32 if not q else 8, gen=60, lam=1.0, out="out/report", skip_fix=False),
           "tree":        dict(n=a.n or (40 if q else 240), steps=a.steps or (60 if q else 400), k=a.k, lam=None, gen=80, sum_tokens=80, voice_budget_ms=a.voice_budget_ms),
           "auction":     dict(n=a.n or (60 if q else 1200), steps=a.steps or (100 if q else 1500), k=a.k, memo_tokens=40, deadline_ms=a.deadline_ms, control_ms=a.control_ms, lam=1.0, seed=a.seed)}
    stages = [s.strip() for s in a.stages.split(",")]
    summary = {}
    t_all = time.time()
    for s in stages:
        print("\n" + "=" * 100 + f"\n STAGE: {s}   sender={a.sender}  receiver={a.receiver}\n" + "=" * 100, flush=True)
        ns = argparse.Namespace(**cfg[s]); t0 = time.time()
        try:
            r = {"latentcheck": run_latentcheck, "tree": run_tree, "auction": run_auction}[s](ns)
            summary[s] = {"ok": True, "minutes": (time.time() - t0) / 60, "result": r}
        except Exception as e:
            import traceback; traceback.print_exc(); summary[s] = {"ok": False, "error": str(e)}
    # ---------------------------------------------------------------- summary
    print("\n" + "#" * 100 + "\n SUMMARY\n" + "#" * 100)
    L = summary.get("latentcheck", {}).get("result")
    if L:
        P = L["probes"]
        print(f"latentcheck ({L['sender']} -> {L['receiver']}): verdict {L['verdict']}")
        print(f"  fidelity {P['fidelity']['value']:.3f} | decay hop{L['hops']}/hop1 {P['decay']['value']:.3f} | format_keep {P['format_keep']['value']:.2f} (text {P['format_keep']['text']:.2f})"
              f" | cue innocent: {P['cue_indep']['pass']} | fix: format {P['fix']['format']:.2f} at -{P['fix']['fidelity_cost']:.3f} fidelity" if 'fix' in P else "")
        print(f"  latency: {P['latency']['fast_hop_ms']:.1f} ms/hop vs {P['latency']['text_decode_ms']:.0f} ms text -> {P['latency']['speedup_per_hop']:.0f}x")
    T = summary.get("tree", {}).get("result")
    if T:
        print(f"tree: text {T['latency_ms']['text_tree_to_decision']:.0f} ms vs latent {T['latency_ms']['latent_tree_to_decision']:.0f} ms ({T['latency_ms']['speedup']:.0f}x); "
              f"runbook format kept: text {T['text_tree']['runbook_format_rate_6']:.2f} latent {T['latent_tree']['runbook_format_rate_6']:.2f}")
        for r, d in T["ablation"].items(): print(f"  drop {r:11s}: d(severity acc) {d['d_severity_acc']:+.3f}  d(runbook CE) {d['d_runbook_ce']:+.2f}")
        if "voice_turn" in T:
            V = T["voice_turn"]; print(f"  voice turn @{V['budget_ms']:.0f} ms: text tree fits {V['text']['in_budget_rate']:.0%} (median {V['text']['median_ms']:.0f} ms) | latent tree fits {V['latent']['in_budget_rate']:.0%} (median {V['latent']['median_ms']:.0f} ms); "
                  + "accuracy vs text " + " ".join(f"{p} {V['survives']['accuracy'][p]:+.2f}" for p in V['survives']['accuracy']) + f"; format {V['survives']['format']:.2f}")
    A = summary.get("auction", {}).get("result")
    if A:
        dl, cl = f"deadline_{int(A['deadline_ms'])}ms", f"deadline_{int(A['control_ms'])}ms"
        print(f"auction (reference surplus/1k: oracle {A['reference_surplus_per_1k']['oracle']:.0f}, constant prior {A['reference_surplus_per_1k']['constant_prior']:.0f}):")
        for lane in ("blind", "text", "latent"):
            r = A["lanes"][lane]; print(f"  {lane:6s} @{int(A['deadline_ms'])}ms miss {r[dl]['miss_rate']:.2f} surplus {r[dl]['surplus_per_1k']:.1f} | @{int(A['control_ms'])}ms surplus {r[cl]['surplus_per_1k']:.1f} | median {r['ms_median']:.0f} ms")
    print(f"\ntotal {(time.time()-t_all)/60:.1f} min. Files: out/report.html (scorecard), out/*.json. Label: one seed, synthetic tasks, frozen models.")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "result"} | ({"result": v["result"]} if v.get("ok") else {}) for k, v in summary.items()}, open("out/summary.json", "w"), indent=1, default=str)

if __name__ == "__main__":
    main()
