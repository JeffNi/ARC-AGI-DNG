# Roadmap

## Phase 1 — Trial-and-Error Motor Learning (COMPLETE)

Brain learns through repeated attempts + reward feedback on simple grid tasks.
Like a rat learning which lever to press. Slow but foundational.

**Status:** Complete (2026-04-18). Key fix: copy_bias reduced from 0.15 to 0.05.

**Results:**
- [x] Solid fill: solves in 6-9 retries (was stuck at 0.444 with high copy_bias)
- [x] Identity: solves in 4-7 retries via depression
- [x] Color swap: solves in 8-12 retries
- [x] Color extract: solves in 7-8 retries
- [x] Eval motor loop fixed (deterministic position cycling)
- [x] KC patterns stable across retries (Jaccard 0.78-0.94)
- [ ] No generalization to unseen instances (expected — Phase 2 goal)

## Phase 2 — Within-Task Generalization

Brain sees N instances of a task type, then solves new instances *faster* —
fewer trials needed because MBON weights carry useful information forward.
Still trial-and-error, but informed by prior experience.

**Goal:** Holdout accuracy >> train accuracy improvement. The brain gets better
at a task type, not just a single instance.

**Sub-goals:**
- [ ] Expand to 4x4 grids (3x3 too few combinations, near-random guessing viable)
- [ ] Learnable attention pointer (decouple from POSITION winner)
- [ ] flip_h task becomes solvable with learnable attention
- [ ] color_swap learning via richer KC representations
- [ ] Cross-position relational features (neighbor context, symmetry)

### 4x4 Grid Expansion

3x3 grids have too few cells to meaningfully test reasoning — a 9-cell grid with
2-3 colors means random guessing gets lucky too often. 4x4 (16 cells) provides
enough combinatorial space that success requires actual learning.

Changes needed:
- SENSORY region: 16 cells × 10 colors = 160 sensory neurons (up from 90)
- POSITION pool: 16 neurons (up from 9)
- GAZE pool: 16 neurons (up from 9)
- L3 (KC) population: likely needs scaling proportionally
- Attention gain cache: 16 positions instead of 9
- Task generators: update to produce 4x4 grids
- Action budget / observe steps: may need tuning for larger grid

Do this AFTER Phase 1 parameters are stable. Changing grid size invalidates all
learned parameters, so we want a solid base first.

### Learnable Attention Pointer

Currently attention = POSITION winner (same cell you're acting on). For spatial
transformations (flip_h, translations, rotations), the brain needs to attend to a
*different* cell than the one it's acting on.

Plan: add a small ATTENTION pool (one neuron per cell) with L3→ATTENTION wiring.
The brain learns which cell to look at given the current L3 pattern. ATTENTION
winner drives KC gain modulation instead of POSITION winner.

Prerequisite: basic per-position learning must work first (identity, color_extract).

### Cross-Position Relational Features

Current sensory encoding is per-cell one-hot. No explicit encoding of relationships
between cells (adjacency, symmetry, color frequency).

Biology handles this through lateral connections in early visual areas (V1
orientation columns, surround suppression). Consider: neighbor color features,
row/column statistics, or letting L1/L2 develop these through plasticity.

## Phase 3 — Demo Observation & One-Shot Learning (THE BIG LEAP)

Brain looks at demo input→output pairs, extracts the transformation rule, and
applies it to the test input without trial-and-error. This is analogical reasoning
— the core of what ARC-AGI actually tests.

**This is not a feature addition. It's a fundamentally different kind of computation.**

The brain needs:
- **Working memory** holding two patterns simultaneously (demo input vs output)
- **Comparison circuit** detecting what changed between them
- **Rule abstraction** compressing the difference into a reusable transformation
- **Rule application** applying the abstraction to novel input

Biologically, this maps to prefrontal cortex function — what insects lack and
primates have. We'd be graduating from insect-scale to cortical-like computation.

**Building blocks from Phase 1-2 that carry forward:**
- Spatial attention → comparing grid positions across demo/test
- MBON system → short-term memory for holding extracted rules
- KC sparse coding → representational substrate for encoding transformations
- Reward system → reinforcing successful rule application

**Sub-goals:**
- [ ] Re-enable demo grid observation
- [ ] Comparison mechanism: brain can detect input→output differences
- [ ] Rule extraction: MBON pattern encodes transformation, not just grid content
- [ ] One-shot test: solve test grid from demos alone, no trial-and-error needed

## Phase 4 — Scaling to Full ARC

Deferred until small-grid reasoning works. No point scaling a system that can't
reason at 4x4.

- Hierarchical attention (attend to region, then cell within region)
- Visual field / foveation for large grids (up to 30x30)
- Convolutional sensory encoding (shared weights across positions)
- Working memory for partial results
- Compositional rule understanding (novel task types from known primitives)

---

## Research Questions / Future Explorations

### Brain Hemispheres

Biological brains (vertebrates and many invertebrates including insects) have
bilateral symmetry — two hemispheres processing slightly different
representations of the same input. This is NOT merely an evolutionary quirk:

- **Redundancy and robustness**: damage to one hemisphere doesn't destroy all
  learned associations
- **Lateralized specialization**: in bees, the right antenna/left hemisphere is
  better at learning odor associations, while the left antenna/right hemisphere
  contributes to long-term memory consolidation
- **Ensemble diversity**: two slightly different sparse codes for the same input
  provide richer representational capacity and better generalization
- **Inter-hemispheric comparison**: cross-talk between hemispheres enables
  comparison-based computation (detecting differences, integrating perspectives)

**Potential implementation:**
- Duplicate L3 (KC population) into two hemispheres with independent random wiring
- Each hemisphere develops its own sparse code for the same sensory input
- MBON readout averages or combines both hemispheric representations
- Cross-hemispheric connections (callosal analog) enable comparison circuits
- Could be critical for Phase 3 (comparing demo input vs output patterns)

**Open questions:**
- At our scale (~800 KCs), is splitting into 2×400 worth the representational
  cost vs keeping 1×800?
- Should hemispheres share plasticity rules but differ in initial wiring?
- When in development should cross-hemispheric connections form?

**Priority:** Low for now. Revisit when Phase 2 generalization hits limits, or
when Phase 3 requires comparison circuits.
