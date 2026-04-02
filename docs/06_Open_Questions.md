# Open Questions, Risks, and Roadmap

---

## 1. Resolved Design Decisions

- **Neuron model:** Continuous activations (not spiking). Simpler math, invest complexity in network structure.
- **Neurogenesis:** No runtime node creation. All nodes from template.
- **Connection strategy:** Overshoot then prune. Matches the biological three-phase model.
- **Template design:** Found through evolutionary search, not hand-designed. The "genome" encodes rules for building the network.
- **What's plastic:** Weights, excitability, leak rates, connections. Nearly everything except node types and region assignments.
- **Learning rule:** Contrastive Hebbian Learning (CHL). Biologically plausible alternative to backprop. Compares free-phase (network's guess) to clamped-phase (correct answer) correlations.
- **Output readout:** Integration-to-bound model with per-cell Winner-Take-All (WTA) in the motor region. Each output cell competitively selects one color.
- **Encoding:** One-hot with 0 for inactive (not -1). Avoids spurious correlations.
- **Observation-based learning:** The network first passively observes all training examples (input+output clamped, no weight changes). Then it attempts the test problem. CHL is applied gently to tune the scaffold, not memorize task-specific mappings.
- **Neuromodulators:** DA (reward/punishment), ACh (encoding/plasticity), NE (arousal/exploration) as global scalar signals that modulate learning dynamics.
- **Mode signaling:** The network distinguishes observation from performance via two naturally occurring signals: (1) neuromodulator context (ACh/NE levels) and (2) whether motor neurons are externally driven or self-generating.
- **Instinct circuit:** Copy pathway (sensory→motor, weight 0.3) provides a baseline identity behavior that learning can override.
- **Structural plasticity:** Both synaptogenesis (new edges between co-active neurons) and pruning (sustained-weakness removal) occur at all developmental phases with varying rates.

---

## 2. Still Open

1. **Can recurrent dynamics discover rules?** The core hypothesis. Can a richly recurrent network with Hebbian learning look at input-output examples and internally discover the transformation rule? This is untested.

2. **The binding problem (unsupervised):** In the supervised approach, we solved binding by presenting one mapping at a time. In the observation-based approach, the network must solve this itself through its own dynamics (internal attention via E/I/M interactions). Can it?

3. **Genome parameterization:** Exactly which parameters define the genome for evolutionary search. Region sizes, connectivity densities, E/I ratios, recurrence depth, memory node count, etc.

4. **Variable grid sizes:** ARC grids range from 1x1 to 30x30. Fixed-size sensory window handles this naturally, but the motor region needs to match the expected output size.

5. **Cross-task transfer:** Reset to seed after each task (clean slate) vs. retain modifications (continual learning).

6. **Active attention:** Can the network eventually control which example to revisit? This would require a control mechanism for directing sensory input.

7. **Surprise modulation:** The simple `surprise = mean(|delta_a|)` scalar for learning rate modulation. How much does this help vs. plain constant-rate Hebbian?

8. **Oscillatory dynamics:** Rhythmic signal modulation for temporal binding. Deferred.

9. **Leak rate adaptation rule:** Preliminary. May need a different formulation.

---

## 3. Known Risks

- **The core hypothesis might fail.** Recurrent Hebbian dynamics may not be sufficient to discover transformation rules from observation alone. This would mean either the learning rule needs augmentation (reward signals, more complex plasticity) or the architecture needs more structure (hand-designed processing stages). Either way, the failure would be informative.

- **Attractor quality:** Network may converge to trivial attractors. *Mitigation:* homeostatic excitability, inhibitory nodes for competition.

- **Wrong attractor:** Convergence doesn't guarantee the right answer. *Mitigation:* evolutionary search over genomes biases the energy landscape.

- **Scaling:** Richly recurrent networks with many nodes may be slow. *Mitigation:* sparse connectivity, efficient dynamics implementation.

- **Threshold sensitivity:** Many hyperparameters without gradient-based tuning. *Mitigation:* include them in evolutionary search.

---

## 4. Roadmap

**Phase 1 (COMPLETE):** Design documents and mathematical framework.

**Phase 2 (COMPLETE):** Minimal Python prototype. Validated the machinery (dynamics, Hebbian, pruning, sleep, save/load) on task 0d3d703e using supervised training. This confirmed the infrastructure works but used an unrealistic training approach.

**Phase 3 (Current):** Observation-based CHL learning with neuromodulators. Implementation complete:
- Richly recurrent template with E, I, M, Mem nodes and instinct copy pathway
- Observation-first paradigm: passive observation builds dynamic state, CHL tunes scaffold
- DA/ACh/NE neuromodulators for mode signaling and learning modulation
- Structural plasticity (synaptogenesis + pruning) across developmental phases
- Numba JIT compilation for ~3.5x speedup
- Variable grid support (up to 30x30, tested at 10x10)
- Currently running 500-day childhood training to assess baseline behavior

**Phase 4 (Next):** Tutorial & Homework system:
- Hand-curated tutorials demonstrating reasoning concepts step by step
- Homework tasks testing transfer of tutorial concepts to novel problems
- Spaced repetition for retention testing
- Punishment signal (DA dip) for persistent failure
- Re-watching tutorials when stuck (with residual context from failed attempts)

**Phase 5:** Evolutionary search over genomes. Find templates where the dynamics can discover rules.

**Phase 6:** Scale to broader ARC task categories. Identify what additional structure (if any) the genome needs.

**Phase 7:** Full ARC-AGI evaluation. Compare against baselines.
