# Related Work

Existing approaches and how the DNG differs.

---

## 1. Architectures With Dynamic Topology

- **NEAT (Stanley & Miikkulainen, 2002):** Evolves network topology via genetic algorithms. *Limitation:* population-based evolution, not signal-driven self-organization.
- **Growing Neural Gas (Fritzke, 1995):** Adds nodes/connections via Hebb-like rule. *Limitation:* unsupervised only, no recurrence.
- **Adaptive Resonance Theory (Grossberg):** Creates new category nodes on demand. *Limitation:* rigid two-layer architecture.
- **SMGrNN (2025):** Structural plasticity module for neuron insertion/pruning. *Most similar* but applied to RL, not recurrent signal-flow reasoning.

---

## 2. Biologically-Inspired Continuous Dynamics

- **Liquid Neural Networks (MIT, 2020-2025):** Continuous-time recurrent networks with input-dependent time constants. Biologically inspired, interpretable, efficient. *Limitation:* fixed topology.
- **Echo State Networks (Jaeger, 2001):** Fixed random reservoir, only output weights trained. *Limitation:* no internal learning or structural change.
- **Modern Hopfield Networks (Ramsauer, 2020):** Energy-based attractor networks with exponential storage capacity. *Our energy framework is inspired by this.*

---

## 3. Non-LLM Approaches to ARC

- **Program Synthesis (Pang, 2025):** Best non-LLM on public sets: 77.1% ARC-AGI-1, 26% ARC-AGI-2. Kaggle private evaluation scores are much lower (~24% top for ARC-AGI-2). Public set scores are inflated by data leakage and extensive tuning. Evolutionary search over DSL compositions. *Strength:* exact symbolic reasoning. *Weakness:* relies on hand-designed DSL primitives.
- **Neural Cellular Automata (UT Austin, 2025):** Local self-organizing update rules on grids. Matches GPT-4.5 performance at a fraction of cost. *Most relevant competitor* -- also bio-inspired, local rules, self-organizing. But fixed topology and grid-native only (not a general reasoning architecture).
- **Test-Time Training (TTT):** Fine-tunes a model on each task's training examples at inference time. Effective but still gradient-based.

---

## 4. LLMs and ARC: The Full Picture

A common misconception is that LLMs "just need better perception" to solve ARC. The reality is more nuanced:

**LLMs have genuine reasoning capabilities** -- they handle deductive logic, step-by-step math, and linguistic inference. But they fail at:
- **Spatial reasoning:** Performance is "largely random" at the instance level (2025 study)
- **Compositional generalization:** Characterized as "shallow disjunctive reasoners" using shortcuts
- **Consistency:** Minor rephrasing changes answers
- They can *explain* algorithms perfectly while *failing to execute them* ("computational split-brain syndrome")

**Multimodal doesn't help:** GPT-4V (vision + LLM) performs *worse* than text-only GPT-4 on ARC. Vision encoders trained on natural images extract useless features for abstract grids.

**None of the ARC benchmarks are solved.** ARC-AGI-1 best private: ~53%. ARC-AGI-2 best private: ~24% (active Kaggle competition). ARC-AGI-3: best AI 0.37%, humans 100%. Public set scores (e.g., 98% on ARC-AGI-1 public) are misleading -- private evaluation on unseen tasks is the real measure.

**The bottleneck is architectural, not computational.** Next-token prediction has a fundamental ceiling for certain types of reasoning. You cannot benchmark-engineer or scale your way past it.

---

## 5. Biological Mechanisms — Key References

Our learning rules and homeostatic systems are grounded in specific experimental findings:

### Synaptic Scaling (Homeostasis)
- **Turrigiano et al. (1998):** "Activity-dependent scaling of quantal amplitude in neocortical neurons." *Nature* 391, 892–896. Discovered multiplicative synaptic scaling — neurons adjust all incoming weights to maintain a target firing rate. Preserves relative weight ratios (learned structure survives).
- **Turrigiano (2008):** "The self-tuning neuron: synaptic scaling of excitatory synapses." *Cell* 135, 422–435. Review establishing that scaling operates on hours-to-days timescale, requires protein synthesis (TNF-alpha, Arc/Arg3.1), and is mechanistically distinct from fast Hebbian plasticity.
- **Turrigiano (2011):** "Too many cooks? Intrinsic and synaptic homeostatic mechanisms in cortical circuit refinement." *Annual Review of Neuroscience* 34, 89–103. Argues homeostatic and Hebbian mechanisms must operate on different timescales to avoid interference — the principle behind our sleep-only homeostasis.

### Sleep and Synaptic Renormalization
- **Tononi & Cirelli (2003, 2006, 2014):** Synaptic Homeostasis Hypothesis (SHY). Waking potentiates synapses; sleep selectively downscales inactive ones. "Sleep and the price of plasticity" (2014), *Neuron* 81, 12–34. Our SHY-based sleep module (`sleep_selective`) implements this directly.
- **De Vivo et al. (2017):** "Ultrastructural evidence for synaptic scaling across the wake/sleep cycle." *Science* 355, 507–510. Electron microscopy confirmed synapses are ~18% larger after wake vs. sleep, validating the SHY prediction.

### Competitive Hebbian Learning
- **Krotov & Hopfield (2019):** "Unsupervised learning by competing hidden units." *PNAS* 116, 7723–7731. Contrastive rule `dW = g(h) * (input - weight)` for competitive pools. Self-normalizing: weights converge to input centroids (like k-means). Our plasticity kernel implements this exactly.
- **DeSieno (1988):** "Adding a conscience to competitive learning." *IEEE ICNN*. Conscience mechanism: neurons that win too often get a raised threshold, forcing rotation of winners across stimuli. Prevents monopolization in competitive pools.
- **Rumelhart & Zipser (1985):** "Feature discovery by competitive learning." *Cognitive Science* 9, 75–112. Original formalization of competitive learning with winner-take-all and weight normalization.

### Structural Plasticity and Critical Periods
- **Hensch (2005):** "Critical period plasticity in local cortical circuits." *Nature Reviews Neuroscience* 6, 877–888. Critical periods are controlled by E/I balance maturation, not a genetic clock alone. Our developmental stages modulate WTA strictness and plasticity rates accordingly.
- **Bhatt et al. (2009):** "Dendritic spine dynamics." *Annual Review of Physiology* 71, 261–282. LTP induces new dendritic spine formation — the biological basis for our co-activity-driven synaptogenesis.
- **Kwon & Bhatt (2022):** New spines form near recently potentiated spines. *Nature Neuroscience*. Supports our model where synaptogenesis candidates are drawn from co-active neuron pairs.

### Retinotopic Position Encoding
- **Hubel & Wiesel (1962, 1968):** V1 neurons have retinotopic receptive fields — they encode *where* a stimulus is, not just *what* it is. Our position features (normalized row/column) are a minimal implementation of this.
- **Sereno et al. (1995):** "Borders of multiple visual areas in humans revealed by functional MRI." *Science* 268, 889–893. Confirmed retinotopic maps in human V1 via fMRI.

---

## 6. What Our Architecture Uniquely Offers

No existing system combines ALL of:

1. A genetically architected seed network with innate computational primitives
2. Overshoot-then-prune structural plasticity at inference time
3. Local learning rules (no backprop) for weights, excitability, time constants, and connectivity
4. Recurrent signal flow as the mechanism for "thinking"
5. Attractor formation as the representation of understanding
6. Native 2D spatial processing (the key bottleneck LLMs cannot solve)
7. Evolutionary search for the template itself
