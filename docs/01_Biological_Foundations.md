# Biological Foundations

The goal is not to simulate the brain in full detail, but to identify **which biological mechanisms matter for reasoning** and translate them into computable rules.

---

## 1. Multi-Timescale Plasticity

The brain operates on multiple timescales of change:

- **~1-100 ms:** Signal propagation and synaptic transmission -> our activation dynamics
- **~100 ms - 10s:** Short-term facilitation/depression -> fast weight modulation
- **~10s - hours:** LTP/LTD via Hebbian/STDP mechanisms -> our weight updates
- **~hours - weeks:** Synaptogenesis and pruning -> our structural plasticity (connection pruning/formation)

**Key insight (2024, eLife):** Homeostatic synaptic scaling and structural plasticity are *co-dependent* -- they dynamically compensate for each other. Our weight and structural plasticity processes must be coupled.

---

## 2. Spike-Timing-Dependent Plasticity (STDP)

STDP adjusts synaptic strength based on relative timing of pre/post-synaptic activity. In recurrent networks, STDP produces **symmetry breaking** -- neuronal groups specialize to different inputs, with within-group connections strengthened at the expense of between-group connections (Biological Cybernetics, 2010). This is exactly the modular structure we want.

**For our architecture:** We abstract STDP into a simpler BCM-Hebbian rule. We don't need spiking dynamics -- the key principle is: *correlation in activation implies causal structure, so strengthen the connection; anti-correlation weakens it.*

---

## 3. Structural Plasticity (Not Neurogenesis)

A critical biological fact: **the brain is born with essentially all its neurons (~86 billion).** Adult neurogenesis is extremely limited (~1,400 neurons/day in the hippocampus at most, and even this is disputed). What changes dramatically after birth is **connections**:

- **Overshoot phase (birth - age ~2):** Massive overproduction of synapses (~2x adult levels)
- **Pruning phase (age 2 - adolescence):** Activity-dependent elimination sculpts the excess into efficient circuits
- **Stabilization (adulthood):** Slow ongoing refinement, limited new connection formation

The three-phase strategy -- **overshoot, prune, stabilize** -- is our structural plasticity model. Not "start sparse, grow as needed."

Connection formation is guided by both activity-independent molecular cues (genetic) and activity-dependent mechanisms. A 2024 Nature Neuroscience paper showed that activity-regulated transcription factors directly drive synaptogenesis -- neural activity turns on genes that build synapses. Growth cones on axon tips follow chemical gradients to find appropriate targets.

**Biphasic structural plasticity (eLife, 2024):** Moderate activity increases connections; extreme activity or silence decreases them. This prevents both runaway growth and total atrophy.

---

## 4. Cell Assemblies and Attractors

Hebb (1949) proposed that co-activating neurons form **cell assemblies** -- stable ensembles representing concepts. Modern evidence:

- **Assembly delineation (bioRxiv, 2025):** Relies on stronger *indirect inhibitory connections* between assemblies, not just excitatory connections within them. We need both E and I connections.
- **Working memory (Nature, 2024):** Mnemonic information persists in *functional connection patterns* during neural "Off" states -- **structure itself encodes memory**, not just activations. This validates our core idea.
- **Intrinsic excitability (Frontiers, 2024):** Neurons change their own responsiveness to participate in ensembles. Each node needs a tunable gain parameter.

---

## 5. What Is Genetic vs. What Is Plastic

This is critical for our design. Research shows (Nature Communications, 2024):

- **Genes control ~33% of functional connectivity differences** between individuals
- Genes primarily shape **topography (spatial organization)** -- WHERE things connect
- **Connection strength is shaped primarily by environment/learning**
- Hub connectivity (most important pathways) is more genetically influenced
- Time constants, intrinsic excitability, and detailed connectivity are all **plastic** (modifiable by experience)
- At birth, basic cortical regions show functional organization but differentiation is "far from complete"
- V1 has plasticity windows extending into the **third decade of life**

**Implication for our DNG:** The genetic template mainly needs to specify *topology* (regions, rough connectivity rules). Nearly everything else -- weights, time constants, excitability, detailed connectivity -- can be learned through plasticity. This makes the template design tractable.

---

## 6. Neural Oscillations and the Binding Problem

The brain coordinates distributed processing through oscillations:

- **Gamma (~40-90 Hz):** Binds features into coherent percepts via synchronous co-ripples across distant regions
- **Theta (~4-7 Hz):** Organizes sequences and coordinates memory encoding
- **Theta-gamma coupling:** Multiple items represented as gamma cycles nested within theta cycles

**For our architecture (future extension):** Oscillatory signal injection could help bind related features and sequence processing steps. Deferred to v2 for simplicity.

---

## 7. Predictive Coding and the Free Energy Principle

Karl Friston's Free Energy Principle: the brain minimizes **surprise** by updating its internal model to better predict sensory inputs. Predictive Coding Networks implement this -- each layer predicts the layer below, only prediction errors propagate upward.

**For our architecture:** The free energy principle provides theoretical justification for our energy function. Our system is a free energy minimizer -- it adjusts activations, weights, and structure to minimize the discrepancy between internal dynamics and injected signals. Feedback connections from higher to lower regions implement the top-down predictions.

---

## 8. Sleep: Global Synaptic Downscaling

Sleep is not idle time -- it is a critical computational phase with direct implications for our architecture.

**The Synaptic Homeostasis Hypothesis (Tononi & Cirelli):**

- During **wakefulness**, synapses strengthen through experience (Hebbian learning). Net synaptic weight increases throughout the day.
- During **slow-wave sleep (SWS)**, synapses are **globally downscaled** -- a bulk normalization. The brain reduces AMPA receptors broadly across the cortex (2024, PLOS Biology).
- Strong connections (well-learned) survive downscaling. Weak ones (noise, irrelevant associations) fall below threshold and vanish.
- This is NOT the same as activity-dependent pruning (our Process 3). Sleep is an indiscriminate, global rescaling that acts on ALL connections simultaneously.

**Memory consolidation during sleep:**

- Coordinated oscillations during non-REM sleep (delta waves, spindles, sharp-wave ripples) replay and transfer information between hippocampus and cortex (Communications Biology, 2024).
- Spindle density predicts increased synaptic consolidation.
- Sharp-wave ripples act as a binary switch triggering memory transfer.

**Sleep pressure is driven by synaptic strength:** The more you learn (the more synapses strengthen), the more you need to sleep. Sleep pressure is literally a measure of accumulated synaptic potentiation (Nature Reviews Neuroscience, 2024).

**For our architecture:** We should add a **sleep phase** to the task processing pipeline -- a step where all weights are globally downscaled by a factor (e.g., multiply all W by 0.8). Strong task-relevant connections survive. Weak incidental connections disappear. This is complementary to activity-dependent pruning and may be critical for generalization (removing overfitting to specific examples while preserving the underlying rule).

---

## 9. Internal Signal Generation: The Brain Is Never Silent

A major gap in our current design: we only inject external signals (ARC grids). But the brain generates massive internal activity at all times.

**The brain is primarily internally driven:**

- Spontaneous activity accounts for **60-80% of total brain metabolic energy** (Communications Biology, 2025). External stimulus processing is the minority use case.
- The brain uses 20% of body energy despite being 2% of body mass, mostly for spontaneous internal activity.
- At the network level, neurons exhibit spontaneous Up-Down state oscillations even without input -- membrane potentials rhythmically transition between active and quiet states.

**Where internal signals come from:**

- **Ion channel stochasticity:** Random ion channel opening/closing occasionally triggers spontaneous firing without any input. This is the lowest-level source.
- **Recurrent network dynamics:** In a recurrently connected network, any residual activation reverberates and sustains itself. The network's own structure generates ongoing activity.
- **Default Mode Network (DMN):** A set of brain regions most active when NOT processing external tasks -- it performs memory replay, planning, and self-referential thinking.

**What this internal activity does (most of it IS functionally important):**

- **Memory replay:** The hippocampus replays past experiences in time-compressed form during rest and sleep via sharp-wave ripples (~150 Hz bursts). A 2024 Nature Neuroscience paper showed these replays function like "policy rollouts" -- the brain simulates possible action sequences to plan future behavior.
- **Compositional memory construction:** Replay builds NEW behaviors by combining existing primitives (Nature Neuroscience, 2025). The brain infers novel solutions without additional external learning. **This is directly relevant to ARC compositional reasoning.**
- **Daydreaming / mind wandering:** Occupies 25-47% of waking cognition. Not idle -- it optimizes memory efficiency and maintains goal focus (PMC, 2024).
- **Implicit learning boost:** A 2024 study found that mind wandering during implicit learning *improves* extraction of hidden probabilistic patterns. The wandering brain is exploring its own attractor landscape.
- **Awake replay:** Happens during behavioral pauses (when you stop and think). Both forward replay (simulation) and reverse replay (reviewing what happened) occur.

**What we probably DON'T need to model:**

- Self-referential thinking, social cognition, emotional regulation (irrelevant for ARC)
- The specific ion channel stochastic mechanism (simple noise suffices)
- Motor planning replay

**What we probably DO need:**

- **A "rest phase"** where the network runs dynamics without external input, allowing internal replay and consolidation
- **Low-level noise** injected into node activations to enable exploration of nearby attractors
- **The insight that "thinking" happens offline:** After seeing ARC training examples, the network should have time to internally process (replay, recombine) before being asked to generate output

This maps to the human experience: you look at the examples, then pause and *think*, and the insight often comes during reflection, not while staring at the data.

---

## 10. Memory Storage: Structural Engrams

Memory is stored as enduring **structural changes** to the network -- validating our core architectural approach.

**Key findings (Science, 2024 -- 3D electron microscopy of engrams):**

- Long-term memory involves formation of **multisynaptic boutons** (one axon forming multiple synapses with different targets), not just strengthening individual connections
- Memory is distributed across **sparse neural ensembles** throughout the brain
- Engrams exist in two states: **active** (during recall) and **dormant** (stored but inactive)
- Engrams are **dynamic** -- they undergo reconsolidation, representational drift, and updating. Not rigid recordings.
- Synaptic potentiation of engram neurons is both **necessary and sufficient** for memory (Communications Biology, 2025)

**For our architecture:** Our model already stores knowledge as structural changes (modified weights, pruned connections). This IS the engram model. The dormant/active distinction maps naturally: when activations are reset between tasks, the structural changes persist (dormant engram). When signal flows through the shaped network again, the engram becomes active. The finding that engrams are dynamic and updateable matches our continuous plasticity approach.

---

## 11. Perception and Reasoning Are Intertwined

Perception is NOT a simple preprocessing pipeline feeding into a reasoning engine. The brain's visual processing involves massive recurrent feedback -- higher areas generate predictions, lower areas compute errors, and this loops until convergence. Conscious perception requires this recurrent processing (~100-200ms).

For ARC, many tasks that seem like "reasoning" are actually sophisticated perception: symmetry detection, Gestalt grouping, figure-ground separation, contour completion. The visual system does these automatically. A rich perception system with native 2D structure may solve a substantial fraction of ARC without explicit "reasoning."
