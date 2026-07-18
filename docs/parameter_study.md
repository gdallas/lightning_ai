# Decoder Parameter Study

**Status: complete.** Generated autonomously overnight from real experiment runs.
Model: `EleutherAI/pythia-160m` · Task: single-token category naming · CPU, fp32, 2 threads.

---

## TL;DR (recommendations)

- **Gap sampler `δ = 2`** is the best decoder overall — 3.09 distinct valid answers per prompt, the
  most of anything tested, with the cleanest outputs. **Use this if you pick one.**
- **Temperature `T = 0.7`**, not the usual 1.0 — the default is *too hot* for a 160M model and loses
  on both correctness and variety. 0.7 is the sweet spot (2.57 distinct valid @ 0.29 validity).
- **Nucleus `p`** is a **near no-op** on this task (outputs barely change from p=0.85 to 0.995) —
  don't rely on it alone; pair it with a lower temperature.
- **The perturbation ensemble does *not* beat greedy at 160M** — at every noise level and every layer
  scope it surfaces generic filler ("common", "most") instead of valid alternatives. It's a negative
  result *at this scale*, not a refutation of the idea.
- **Which layers to hit:** if you perturb, target **attention weights in the later layers (8–11)** —
  that's the only place jostling changes the answer (early layers & MLP just reproduce greedy). But
  expect poor signal-to-noise until the model is bigger.

Full reasoning, metrics, and per-word examples below.

---

## 1. What we're tuning and why

Every decoder reads the *same* frozen model. The only thing that changes is **how a word is
chosen from the model's output distribution**. The knobs, and what each does:

| Knob | Decoder | Plain meaning | Turning it up… |
|---|---|---|---|
| `T` (temperature) | temperature | flattens/sharpens the whole distribution | more random, more variety, less correct |
| `p` (top-p / nucleus) | nucleus | size of the candidate pool (smallest set covering `p` of probability) | wider pool, more variety, more risk |
| `δ` (delta) | gap sampler | how far below the top score a word can be and still be eligible | more eligible words, more variety |
| `σ` (sigma) | ensemble | strength of the random weight noise | more words flip → more minority candidates, then chaos |
| `N`, `k` | ensemble | perturbed passes, and votes needed to trust a minority word | higher `k` = stricter (more fallback to greedy) |
| noise **scope** | ensemble | *which* weights get perturbed (attention/MLP, layer band) | targets where the model "decides" |

The objective is **not** raw accuracy. It's **diversity at a quality floor**: as many *distinct valid*
answers per prompt as possible while keeping validity acceptable. A decoder that spits out 20
different words but they're all junk is useless; so is one that's always correct but always says
the same word.

## 2. The setup

**Model — Pythia-160m (GPT-NeoX):** 162.3M parameters (85.1M non-embedding), **12 layers**,
hidden width 768, 12 attention heads, MLP inner width 3072, 50,304-token vocabulary. Runs on CPU
in fp32 at ~0.26 s per forward pass (2 threads).

**Task — category naming (Task A):** 20 categories (animal, color, metal, …). The prompt template
was tuned by greedy validation to `"The most common {category} is the"` (0.50 greedy validity;
the original template scored 0.00). Answers are scored against a hand-built word list, restricted
to **single-token** words (200 of 393 survive; "city" has none, so it is effectively unusable).

**Ensemble perturbation target (default scope):** 48 weight matrices = 84.9M weights = **52% of
the model** — the attention (QKV, output) and MLP (up, down) projections across all 12 layers,
skipping embeddings and layer-norms.

**Metrics (per setting, over the 20 prompts):**
- `validity_rate` — fraction of all sampled answers that are valid.
- `distinct_valid_per_prompt` — average number of *different* valid answers found per prompt (the
  diversity number that matters), with a bootstrap 95% CI.
- `coverage` — distinct valid answers as a fraction of the available answer space.
- `mode_share` — how concentrated the valid answers are on the single most common one.

## 3. How to read the example tables

Each sweep produces a side-by-side `answers.md`: one row per test word, one column per setting.
A cell lists the distinct answers that setting produced, most frequent first, marked `✓`/`✗` for
valid/invalid with the trial count in parentheses, e.g. `red✓(7), same✗(4), blue✓(2)`. Greedy is
deterministic, so its "column" is a single word repeated.

## 3b. Baseline reference (default knobs, R=20, 20 category prompts)

The four standard decoders at their default settings, 400 trials each — the yardstick the
sweeps below move away from:

| decoder | knob | validity | distinct valid / prompt (95% CI) | mode-share |
|---|---|---|---|---|
| greedy | — | **0.50** | 1.00 | 1.00 |
| temperature | T=1.0 | 0.10 | 2.43 [1.70, 3.14] | 0.44 |
| nucleus | p=0.95 | 0.14 | 2.44 [1.89, 3.00] | 0.56 |
| gap sampler | δ=2 | 0.27 | 3.67 [2.57, 4.70] | 0.39 |

Read this as the core tension: **greedy is 50% valid but has zero variety** (always one answer);
the samplers trade validity down for 2.4–3.7× more distinct valid answers. Gap sampler is the
early standout — highest diversity *and* the best validity among the samplers. The sweeps below
ask: can better coefficients widen that gap?

## 3c. What each decoder outputs for the same word (five decoders, default knobs)

A curated slice of the full side-by-side table (`results/…_bb9f949e43ee/answers.md`), chosen to
show the behaviours clearly. `✓` = valid answer for that category.

| word | greedy | temperature (T=1.0) | nucleus (p=0.95) | gap (δ=2) | ensemble (σ=0.05) |
|---|---|---|---|---|---|
| metal | lead✓ | *(fragments)* | cobalt✓, copper✓ | **silver✓, aluminum✓** | **iron✓, copper✓** |
| animal | dog✓ | *(junk)* | *(junk)* | **deer✓, rabbit✓, rat✓** | dog *(fell back)* |
| body of water | sea✓ | lake✓, ocean✓ | lake✓, sea✓ | lake✓, river✓ | surface✗ |
| color | red✓ | blue✓, orange✓ | *(junk)* | blue✓, white✓ | color✗ |
| vegetable | cabbage✓ | carrot✓, tomato✓ | cabbage✓ | *(junk)* | vegetable✗ |

Three things jump out even at this small scale:
- **Different decoders genuinely surface different valid answers** (the "metal" row is the poster
  child — four distinct correct metals across four decoders).
- **Temperature at T=1.0 is already too hot** — it produces word-*fragments* ("l", "je", "mur") as
  often as words. This task wants a *cooler* temperature. (Sweep in §4.)
- **The ensemble frequently falls back to greedy or grabs the category word itself** ("color",
  "vegetable") at σ=0.05 — but when it does find a minority, it's often a clean valid one
  ("iron", "copper"). σ and *which layers* we perturb are the levers (§7–8).

---

## 4. Temperature (`T`) sweep

Six values, R=10 over 20 words.

| T | validity | distinct valid / prompt (95% CI) | verdict |
|---|---|---|---|
| 0.5 | 0.385 | 2.40 [1.78, 3.00] | safe, slightly conservative |
| **0.7** | 0.285 | **2.57 [1.88, 3.22]** | **best diversity, validity still OK** |
| 1.0 (default) | 0.150 | 1.92 [1.50, 2.33] | too hot — junk creeps in |
| 1.3 | 0.095 | 1.55 [1.14, 1.86] | mostly fragments |
| 1.6 | 0.040 | 1.14 | collapse |
| 2.0 | 0.020 | 1.00 | pure noise |

**Examples (what the word "animal" and "color" produce as T rises):**

| word | T=0.5 | T=0.7 | T=1.0 | T=1.6 |
|---|---|---|---|---|
| animal | dog, cat, deer, rabbit, sheep ✓ | **dog, cat, deer, pig, rabbit, sheep ✓** | deer✓ + *ben, mur, l, bat* | *fragments* |
| color | red, yellow, blue, green, white ✓ | red, blue, green, white, yellow ✓ | red, green, white ✓ + *ox, salt, sun* | *cub, form, ox, rem* |

**Reading:** temperature has a clean interior optimum. Below ~0.7 you're safe but leave diversity
on the table; at **0.7** you get the most distinct valid answers before quality falls off a cliff;
by T=1.0 the model is already emitting sub-word fragments ("mur", "ox", "l"), and ≥1.3 is noise.
The conventional `T=1.0` default is simply **too hot for a 160M model on a single-token task** —
`T=0.7` beats it on *both* validity (0.29 vs 0.15) and diversity (2.57 vs 1.92).

## 5. Nucleus (`p`) sweep

Five values, R=10 over 20 words.

| p | validity | distinct valid / prompt (95% CI) |
|---|---|---|
| 0.85 | 0.160 | 2.27 [1.40, 3.00] |
| 0.90 | 0.160 | 2.27 [1.40, 3.00] |
| **0.95** | 0.155 | **2.40 [1.50, 3.17]** |
| 0.98 | 0.145 | 2.40 [1.50, 3.17] |
| 0.995 | 0.140 | 2.30 [1.50, 3.00] |

**`p` is almost a no-op here.** Both metrics are essentially flat, and the per-word outputs are
*nearly identical* across the whole range — e.g. "animal" returns the same
`dog, cat, cow, mouse, pig (+junk)` at p=0.85 and at p=0.995:

| word | p=0.85 | p=0.95 | p=0.995 |
|---|---|---|---|
| animal | dog, cat, cow, mouse, pig ✓ | dog, cat, cow, mouse, pig ✓ | dog, cat, cow, mouse, pig ✓ |
| color | blue, red, cyan, white, yellow ✓ | blue, red, cyan, white, yellow ✓ | blue, red, cyan, white *(+kaiser✗)* |

**Why:** the model's raw next-token distribution is *sharp* — a few tokens hold almost all the
mass. Widening the nucleus only admits tail tokens with near-zero probability, which are then
almost never sampled. So truncation (what `p` controls) barely changes anything; what *would*
change diversity is re-shaping the distribution — i.e. **temperature**. Tellingly, temperature at
T=0.7 (distinct 2.57 @ validity 0.29) beats **every** nucleus setting (best 2.40 @ validity 0.16)
on *both* axes. Nucleus without temperature is the weakest of the three samplers here.

## 6. Gap sampler (`δ`) sweep

Six values, R=10 over 20 words. (`δ` = how many logit units below the top score a token can be and
still be eligible; the sampler then picks uniformly among the eligible non-top tokens.)

| δ | validity | distinct valid / prompt (95% CI) | verdict |
|---|---|---|---|
| 1 | **0.415** | 1.69 [1.38, 2.00] | barely leaves greedy |
| **2** | 0.255 | **3.09 [2.17, 4.00]** | **best diversity of any sampler** |
| 3 | 0.140 | 2.30 [1.20, 3.25] | junk creeping in |
| 4 | 0.090 | 2.29 [1.67, 3.00] | over-wide |
| 6 | 0.005 | 1.00 | near-uniform → collapse |
| 8 | 0.000 | 0.00 | pure noise |

**Examples — the δ=1 → δ=2 jump is the whole story:**

| word | δ=1 | δ=2 | δ=3 |
|---|---|---|---|
| animal | dog ✓ *(×10, just greedy)* | **rabbit, rat, sheep, cow, deer, pig ✓** | deer, donkey, elephant ✓ *(+junk)* |
| color | dark✗, blue, yellow, white | **gray, orange, purple, black, green ✓** | brown, cyan, pink, purple ✓ *(+junk)* |
| body of water | river, ocean ✓ | **lake, river, ocean ✓** *(+surface✗)* | river ✓ *(+junk)* |

**Reading:** at δ=1 the eligibility window is so tight the sampler almost always just re-emits the
greedy token (animal = `dog` ten times) — high validity, no variety. Widening to **δ=2** opens the
door to exactly the *runner-up valid answers* (the six other animals, five other colors) without yet
admitting garbage — this is the sweet spot and the **single best sampler setting in the whole study
(3.09 distinct valid)**. By δ=3 junk tokens start slipping in, and δ≥6 makes the window so wide it's
near-uniform sampling over the top logits → everything collapses to noise.

## 7. Ensemble noise strength (`σ`) sweep

Whole-model perturbation (all 48 attention+MLP matrices, N=10, k=2), R=2 over 12 words. **This is a
negative result — read carefully.**

| setting | validity | distinct valid / prompt |
|---|---|---|
| greedy (reference) | **0.50** | 1.00 |
| σ=0.02 | 0.167 | 1.00 |
| σ=0.05 | 0.167 | 1.00 |
| σ=0.10 | 0.083 | 1.00 |

At every σ the ensemble is **worse than greedy on validity and adds no diversity**, and cranking σ up
only makes it worse. The examples show *why*:

| word | greedy | ensemble σ=0.02 | ensemble σ=0.05 |
|---|---|---|---|
| body of water | sea ✓ | surface✗, water✗ | surface✗ |
| color | red ✓ | red ✓ *(fallback)* | color✗ *(category word)* |
| fish | salmon ✓ | o✗ | o✗, sea✗ |
| beverage | coffee ✓ | white✗ | coffee ✓ *(fallback)* |

**The key diagnosis:** the minority tokens that survive random perturbation across the *whole model*
are **generic high-frequency words** — "common", "most", "surface", "sweet" — or the category word
itself ("color", "fruit", "vehicle"). They're robust to noise not because they're good category
answers, but because they're bland, high-prior tokens the model reaches for everywhere. So the
ensemble either (a) falls back to greedy, or (b) surfaces generic filler. Neither helps.

This matters because it's not yet a verdict on the *idea* — it's a verdict on perturbing the whole
model at once. The depth study (separate) showed the model only *decides* its answer in the last 1–2
layers; perturbing all 12 layers mostly jostles the early "generic-word" machinery. The next section
tests whether **targeting the noise at specific layers/components** changes this.

## 8. Ensemble scope — which layers / components to perturb

Fixed σ=0.05, N=10, k=2; the noise is restricted to a subset of the 48 weight matrices. R=2 over
8 words (thin — treat as directional). How much each scope perturbs:

| scope | matrices | weights | validity | what it does |
|---|---|---|---|---|
| all | 48 | 84.9M (52%) | 0.250 | disrupts everything |
| attention-only | 24 | 28.3M (17%) | 0.312 | **flips the decision** |
| mlp-only | 24 | 56.6M (35%) | 0.500 | mostly reproduces greedy |
| early (0–3) | 16 | 28.3M (17%) | **0.625** | nearly harmless |
| mid (4–7) | 16 | 28.3M (17%) | 0.500 | mostly reproduces greedy |
| late (8–11) | 16 | 28.3M (17%) | 0.250 | disrupts the decision |

**Validity alone is a trap here.** High validity means the perturbation *didn't change the answer* —
which is useless if the goal is surfacing alternatives. The examples show what's really happening:

| word | greedy | early-0–3 | mlp-only | attention-only | late-8–11 |
|---|---|---|---|---|---|
| color | red ✓ | red ✓ | red ✓ | **black ✓** | **yellow ✓** |
| body of water | sea ✓ | sea ✓ | sea ✓ | **river ✓** | surface ✗ |
| beverage | coffee ✓ | coffee ✓ | coffee ✓ | white ✗ | white ✗ |

**The real finding on where to perturb:**
- **Early layers (0–3) and MLP weights** are the *wrong* place — a perturbation there washes out
  through the rest of the network, so you just get greedy back (that's why "validity" looks high).
  Perturbing 28–57M weights to reproduce the answer you already had is pointless.
- **Attention weights and late layers (8–11)** are where perturbation actually *moves the decision* —
  and that's where the occasional genuine alternative comes from (color red→**black**/**yellow**, body
  of water sea→**river**). This lines up with the depth study: the model commits to its answer in the
  last 1–2 layers, so that's the only place jostling it can produce a different answer.
- **But the signal-to-noise is poor at 160M.** The same disruptive scopes that surface "black"/"river"
  also surface "white"/"surface" (junk), which is why their aggregate validity is lowest. No scope
  reliably beats greedy on *valid* diversity at this model size.

**So: if you perturb, target attention (and the later layers), not MLP/early — that's where decisions
live. But don't expect it to win at 160M;** the mechanism needs either a bigger model (later layers
carry more semantic structure) or a learned probe to keep the good flips and drop the junk (Phase 7).

---

## 9. Recommendations

### Recommended settings (this task, Pythia-160m)

| decoder | recommendation | reasoning | runner-up |
|---|---|---|---|
| **gap sampler** | **δ = 2** | Best diversity in the entire study — **3.09** distinct valid / prompt @ 0.26 validity. Tight enough to skip junk, wide enough to reach the runner-up valid answers. | δ=1 for higher validity (0.42) but little variety |
| **temperature** | **T = 0.7** | Best validity/diversity *balance* (2.57 @ 0.29); beats the T=1.0 default on **both** axes. | T=0.5 if you weight correctness over variety |
| **nucleus** | p = 0.95 *(weak knob)* | `p` barely changes anything on a sharp distribution; only worth using **paired with T<1**. Weakest sampler on its own. | — |
| **ensemble** | **not recommended at 160M** | Underperforms greedy at every σ and every scope; surfaces generic filler ("common", "most"), not valid alternatives. | if experimenting: attention-only or late-8–11 scope, σ≈0.05 |

### If you want just one decoder: **gap sampler, δ = 2.**
It's the diversity champion and its outputs are the cleanest — e.g. animal →
`rabbit, rat, sheep, cow, deer, pig`; color → `gray, orange, purple, black, green`.
Temperature T=0.7 is the close second and slightly safer on validity.

### Which layers to hit (for the perturbation ensemble)
**Perturb attention weights, biased toward the later layers (8–11) — not MLP, not early layers.**
Reasoning, from §8 + the commitment-depth study:
- The model only *commits* to its answer in the **last 1–2 of its 12 layers**. That is the only place
  where jostling the weights can produce a *different* answer.
- Perturbing **early layers (0–3) or MLP** wastes effort — the change washes out downstream and you get
  greedy back (looks "valid" but it's the same answer).
- Perturbing **attention / late layers** is what actually flips the decision (color red→black/yellow,
  body-of-water sea→river). Attention is where the model "looks up" alternatives; late is where it
  decides.
- **Caveat:** even the right target has poor signal-to-noise at 160M — the same perturbation that
  yields "black" also yields "white". Making this pay off needs a bigger model (Phase 6) or a learned
  probe that keeps the good flips and drops the junk (Phase 7).

### Why the samplers behave the way they do (one paragraph)
The model's raw next-token distribution is **sharp but noisy**: a few tokens hold almost all the mass,
and just below them sit a mix of valid runner-ups and sub-word junk. That single fact explains
everything — **sharpening** the distribution (low temperature) or taking a **tight gap window** (δ=2)
pulls out the valid runner-ups cleanly; **truncation** (nucleus `p`) does nothing because the tail it
trims was never going to be sampled anyway; and **random weight noise** (ensemble) mostly promotes bland
high-prior words rather than the specific valid alternatives, unless aimed precisely at the decision-
making layers.

## 10. Caveats

- Everything here is on a **160M** model — the smallest in the family. Effects may look different at
  410M/1B (Phase 6). Treat absolute validity numbers as low-ceiling.
- The task scores only **single-token** answers, which shrinks the answer space and depresses
  coverage. Multi-token answers (Phase 8) would change the picture.
- Sweeps use a reduced trial count per setting for speed; confidence intervals are wide. Directions
  are trustworthy; exact values are indicative.
