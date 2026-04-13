# Dynamic Neural Graph (DNG)

> A research project exploring whether intelligence can emerge from structure and plasticity rather than scale and gradient descent.

**Status: Active research — infancy and sensory development complete, entering learning phase.**

---

## The Problem

Modern LLMs have more parameters than the human brain has synapses, yet they fail at puzzles a child solves effortlessly. [ARC-AGI](https://arcprize.org/) quantifies this gap: humans score ~100% on tasks where the best AI systems score 0.37% (ARC-AGI-3, interactive).

The issue isn't scale — it's architecture. LLMs are feedforward pattern matchers with no recurrence, no persistent state, and no way to reason through a novel problem at inference time. Every capability must be baked in at training time.

This project asks: **what if we built something closer to how brains actually work?**

## What Is a Dynamic Neural Graph?

A recurrent network whose structure mutates at runtime. Instead of training a fixed architecture via gradient descent, the DNG:

- Starts from a **genetically architected seed network** — like how the genome specifies brain wiring before birth
- **Grows and prunes connections** based on neural activity — active pathways strengthen, unused ones die
- Uses **local, biologically-plausible learning rules** — Hebbian plasticity, not backpropagation
- Develops through **life stages** (infancy → childhood → adolescence) with smoothly transitioning parameters
- Forms stable activation patterns (**attractors**) that represent learned concepts

The architecture follows the **bee brain** — the simplest biological system known to perform abstract rule learning, delayed matching-to-sample, and transfer to novel stimuli. We build the bee first, then extend with mammalian-style reasoning.

## Architecture

```
SENSORY (2828 nodes)
    │
    ▼
L1 / Optic Lobe (900 neurons) ─── Feature detection, columnar WTA
    │
    ▼
L2 / Projection Neurons (300) ─── Intermediate features, trace-rule invariance
    │
    ▼
L3 / Mushroom Body (800 neurons) ─ Expansion recoding, sparse pattern separation
    │                                 APL global inhibition → 10% sparseness
    ▼
MOTOR / MBONs (1000 nodes) ─────── Output decoding, DA-gated associative learning

Also: MEMORY (100), GATE, GAZE, Copy pathway (SENSORY → MOTOR, decays with maturation)
```

| Layer | Neurons | Biological Analogue | Function |
|-------|---------|---------------------|----------|
| L1 | 900 | Bee optic lobe / Mammalian V1 | Local feature detectors via columnar competition |
| L2 | 300 | Projection neurons / V2-V4 | Temporal binding, modest transformation invariance |
| L3 | 800 | Mushroom body (Kenyon cells) | Expansion recoding with sparse random connectivity (~15 inputs per neuron) |
| MOTOR | 1000 | MBONs / Motor cortex | Per-cell output with WTA color selection |

**Key mechanisms:**
- **APL inhibition** — A single global inhibitory signal (modeled on the bee's APL neuron) enforces ~10% sparseness in L3, creating high-dimensional sparse codes for pattern separation
- **Developmental staging** — Parameters interpolate smoothly between life stages (WTA strictness, plasticity rates, inhibitory balance) rather than switching abruptly
- **Contrastive Hebbian Learning (CHL)** — Error-driven weight updates without backpropagation. The difference between "free" and "clamped" network states provides a local error signal
- **Sleep consolidation** — DA-tagged replay of surprising experiences, SHY-hypothesis downscaling of synaptic weights

## Current Progress

### Completed: Unsupervised Development (40-day lifecycle)

The network successfully develops stable representations through infancy and late infancy:

**L1 (Feature Detection) — Stable**
- ~220 alive neurons with selectivity ~0.21
- Column switching 97-99/100 — reliably produces different activation patterns for different inputs
- No collapse despite 40 days of continuous plasticity

**L2 (Intermediate Features) — Functional**
- Discriminates confusable patterns that L1 cannot (L2 vs L1 improvement: +0.44 on confusable pairs)
- Driven fraction ~0.42 at maturity

**L3 (Mushroom Body) — Working as designed**
- Sparseness locked at exactly 10% for all 40 days (APL inhibition working)
- Category separation (catSep) rose from 0.13 at birth to ~0.60 — different inputs activate different sparse subsets
- 150+ unique active neurons per probe

**Gaze System — Developing**
- Fixation events grew from 7 (Day 5) to 37 (Day 40)
- Motor events: 11 → 48
- Spatial scanning emerged (adjacent transition ratio well above chance)
- Stimulus-directed attention: 21/37 fixations on stimulus by Day 40

**Infrastructure**
- 1.2M edges at maturity (1.25x growth from 987K at birth)
- Weights continuously sculpted (w90: 0.183 → 0.146)
- Full run: ~69 minutes on consumer CPU

### Previously Completed
- Identity task mastery (Day 43, 100% solve rate from copy pathway)
- CHL producing immediate weight corrections on failures
- Graded per-cell dopamine-modulated reward (2x larger updates on errors)
- DA-tagged sleep replay (surprising experiences first)
- Rule-based verification (checks transformation rule, not pixel matching)
- 156 regression tests across graph, plasticity, kernels, perception, and brain engine

## Roadmap

### Phase 1: Bee Brain (current)

Single-rule pattern learning via mushroom body → MOTOR associations.

- [x] L1 feature learning — unsupervised local detectors
- [x] Developmental staging — infancy through late infancy with smooth transitions
- [x] Mushroom body (L3) — expansion recoding, APL inhibition, sparse pattern separation
- [x] Gaze system — active visual exploration of stimuli
- [x] Identity mastery via copy pathway
- [ ] **Childhood supervised learning** — CHL + eligibility-modulated reward on curriculum tasks
- [ ] **L3 → MOTOR association learning** — DA-gated KC→MBON plasticity for rule encoding
- [ ] **Multi-task curriculum** — solid fill, binarize, color swap, flip without catastrophic interference
- [ ] **Wire GPU acceleration** — CuPy kernels exist, need integration for 2000-neuron network
- [ ] **Benchmark on ARC-AGI-1 training set**

### Phase 2: Reasoning Module (planned)

Add mammalian-style working memory for multi-step compositional reasoning on top of L3's sparse pattern codes. This follows the evolutionary sequence — insects evolved mushroom bodies ~500M years ago, mammals later evolved neocortex on top of similar pattern separation machinery.

- [ ] Working memory via MEMORY + GATE regions
- [ ] Sequential rule application (compose transformations)
- [ ] Benchmark on full ARC-AGI-1, then ARC-AGI-2

## Why This Approach?

**Biological plausibility over benchmark optimization.** The brain is the only known system that does what we're trying to do — flexible abstract reasoning from minimal examples. Rather than engineering novel architectures, we copy biological mechanisms and adapt them to our scale (~4000 neurons vs 86 billion). Unlike a real brain, we do not need neurons to handle irrelevant tasks such as emotion, temperature detection, etc.

**Bee-first, mammal-later.** Bees demonstrate abstract rule learning, delayed matching, and transfer to novel stimuli with ~1 million neurons. Our 4000-neuron budget is closer to a bee than a mouse (~70M neurons). Build what biology proves is sufficient, then extend.

**Development, not just training.** Biological brains aren't randomly initialized and gradient-descended into competence. They develop through critical periods with changing plasticity rules, E/I balance, and connectivity patterns. We model this developmental trajectory explicitly.

## Hardware

Runs on consumer hardware. The full 40-day developmental lifecycle completes in ~70 minutes on CPU. GPU acceleration (CuPy kernels, written but not yet integrated) will reduce this further.

## Project Structure

```
src/
  brain/          # Engine (dynamics loop), sleep, neuromodulators
  homeostasis/    # Developmental stages, setpoints, stage transitions
  graph.py        # DNG data structure (nodes, edges, regions)
  numba_kernels.py # JIT-compiled inner loops (dynamics + plasticity)
  gpu_kernels.py  # CuPy GPU kernels (written, not yet wired)
  template.py     # Genetic scaffold (initial wiring)
  genome.py       # Hyperparameters
  encoding.py     # Grid ↔ signal conversion
  plasticity.py   # Learning rules (CHL, Hebbian, eligibility, pruning)
  teacher.py      # Curriculum, task presentation, reward
  rule_verifiers.py # Transformation verification
docs/             # Design documents (biology, architecture, math, strategy)
journal/          # Development log (what worked, what didn't, why)
life/             # Checkpoints from developmental runs
tests/            # 156 regression tests
```

## Design Documents

| Document | Contents |
|---|---|
| [Biological Foundations](docs/01_Biological_Foundations.md) | Neuroscience research informing the design |
| [Architecture](docs/02_Architecture.md) | Genetic template, regions, circuit motifs |
| [Mathematics](docs/03_Mathematics.md) | Activation dynamics, plasticity rules, energy function |
| [ARC Strategy](docs/04_ARC_Strategy.md) | Signal encoding, task pipeline |
| [Related Work](docs/05_Related_Work.md) | NEAT, Liquid NNs, NCA, program synthesis, LLM analysis |
| [Open Questions](docs/06_Open_Questions.md) | Design decisions, risks, roadmap |

## ARC-AGI Benchmark Context

| Benchmark | Best AI Score | Human Score | Status |
|-----------|---------------|-------------|--------|
| ARC-AGI-1 | ~53% (Kaggle 2024) | ~95% | Unsolved at human level |
| ARC-AGI-2 | ~24% (Kaggle 2025) | ~95% | Far from solved |
| ARC-AGI-3 | 0.37% | 100% | Interactive, barely started |

We target ARC-AGI-1 first (same format as 2, simpler tasks, faster iteration).
