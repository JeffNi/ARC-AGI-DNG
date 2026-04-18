# Dynamic Neural Graph (DNG)

> A research project exploring whether intelligence can emerge from structure and plasticity rather than scale and gradient descent.

**Status: Active research — Phase 1 complete (trial-and-error motor learning). Entering Phase 2 (within-task generalization).**

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
SENSORY (90 neurons: 9 cells × 10 colors)
    │
    ▼
L3 / Mushroom Body (800 Kenyon Cells)
    │   Local receptive fields (kc_local_fan per assigned cell)
    │   + random cross-cell context (kc_global_fan)
    │   APL global inhibition → ~10% sparseness
    ▼
MEMORY / MBONs (100 neurons: 10 groups × 10 colors)
    │   Depression-only learning (L3→MEMORY)
    │   Readout injected into ACTION with cross-group inhibition
    ▼
ACTION (90 neurons: 9 cells × 10 colors)
    │   copy_bias + MBON drive + observed-color gating
    │   Deterministic position cycling (teacher-driven)
    ▼
Output grid (3×3)

Also: GAZE (9), POSITION (9), DONE, COMMIT
```

| Region | Neurons | Biological Analogue | Function |
|--------|---------|---------------------|----------|
| SENSORY | 90 | Photoreceptors | One-hot color encoding per grid cell |
| L3 | 800 | Mushroom body (Kenyon cells) | Expansion recoding with local receptive fields |
| MEMORY | 100 | MBONs | 10 color-specific groups; depression encodes "not this color" |
| ACTION | 90 | Motor neurons | Per-cell color selection via WTA |

**Key mechanisms:**
- **Local KC wiring** — Each Kenyon cell has an assigned grid position and samples `kc_local_fan` sensory neurons from that cell plus `kc_global_fan` random cross-cell inputs, creating position-specific sparse codes
- **Attentional spotlight** — Center-surround gain modulation amplifies the target cell's sensory signal and suppresses surround, with L3 state reset before each commit
- **Depression-only learning** — L3→MEMORY weights only decrease (wrong-color groups get depressed proportional to KC firing). No potentiation. Recovery during sleep drifts weights back toward baseline
- **copy_bias** — Innate additive bias for the observed input color (identity reflex), kept low enough (0.05) that depression can override it within a few retries
- **Observed-color gating** — Action selection restricted to colors present in the input grid
- **Developmental staging** — Parameters interpolate smoothly between life stages rather than switching abruptly
- **Sleep consolidation** — SHY-hypothesis downscaling; recovery_rate drifts depressed MBON weights back toward initial values

## Current Progress

### Phase 1 Complete: Trial-and-Error Motor Learning

The brain learns to solve simple grid transformations through repeated attempts and reward feedback — analogous to a rat learning which lever to press, or a bee learning which color leads to sugar water.

**Tasks solved (3×3 grids, 4-12 retries per instance):**
- Identity — copy input to output
- Solid fill — fill entire grid with a single color
- Color swap — replace color A with color B everywhere
- Color extract — output only cells matching a target color

**How it works:**
1. Brain observes input grid via SENSORY neurons
2. For each cell position, L3 Kenyon cells fire a sparse pattern
3. MBON readout + copy_bias produce an initial color guess (usually the input color)
4. Wrong guesses trigger depression of the active KC→MBON connections for that color group
5. On retry, the depressed pathway is weaker, so a different color wins
6. Repeats until correct (or retry budget exhausted)

**Known limitations:**
- No generalization to unseen instances — depression is instance-specific
- Spatial transformations (flip, rotate) not solvable without learnable attention
- Learning is purely eliminative (suppress wrong answers), not associative

**Key metrics:**
- KC patterns stable across retries (Jaccard 0.78–0.94)
- APL inhibition maintains ~10% L3 sparseness
- Eval motor loop: deterministic 9-position cycling, matching training

## Roadmap

### Phase 1: Bee Brain — Trial-and-Error Learning (COMPLETE)

Single-rule pattern learning via mushroom body → MBON depression.

- [x] Mushroom body (L3) — expansion recoding, APL inhibition, local receptive fields
- [x] MBON depression-only learning — suppress wrong colors across retries
- [x] Attentional spotlight — center-surround gain with L3 reset
- [x] copy_bias tuning — low enough for depression to override (0.05)
- [x] Deterministic motor loop — teacher-driven position cycling
- [x] Multi-task curriculum — identity, solid_fill, color_swap, color_extract
- [x] Evolutionary parameter search (evolve.py)

### Phase 2: Within-Task Generalization (current)

Brain sees N instances of a task type, then solves new instances *faster*. MBON weights carry useful information forward between instances.

- [ ] Expand to 4×4 grids (3×3 too few combinations)
- [ ] Learnable attention pointer (decouple from POSITION winner)
- [ ] Cross-position relational features
- [ ] flip_h solvable with learnable attention
- [ ] Holdout accuracy >> random baseline

### Phase 3: Demo Observation & One-Shot Learning

Brain observes input→output demo pairs and extracts the transformation rule. Analogical reasoning — the core of what ARC-AGI tests.

- [ ] Re-enable demo grid observation
- [ ] Comparison circuit for input→output differences
- [ ] Rule extraction via MBON patterns
- [ ] One-shot test solving from demos alone

## Why This Approach?

**Biological plausibility over benchmark optimization.** The brain is the only known system that does what we're trying to do — flexible abstract reasoning from minimal examples. Rather than engineering novel architectures, we copy biological mechanisms and adapt them to our scale (~4000 neurons vs 86 billion). Unlike a real brain, we do not need neurons to handle irrelevant tasks such as emotion, temperature detection, etc.

**Bee-first, mammal-later.** Bees demonstrate abstract rule learning, delayed matching, and transfer to novel stimuli with ~1 million neurons. Our 4000-neuron budget is closer to a bee than a mouse (~70M neurons). Build what biology proves is sufficient, then extend.

**Development, not just training.** Biological brains aren't randomly initialized and gradient-descended into competence. They develop through critical periods with changing plasticity rules, E/I balance, and connectivity patterns. We model this developmental trajectory explicitly.

## Hardware

Runs on consumer hardware. GPU acceleration via CuPy kernels (written, not yet integrated) will reduce iteration time further.

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
