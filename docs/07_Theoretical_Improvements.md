# Theoretical Improvements for Continuous Learning

Analysis of the DNG architecture through the lens of non-convergent continuous learning, with concrete suggestions grounded in theory.

---

## 1. Symmetrize CHL Weight Pairs

**Problem:** CHL's theoretical guarantee (gradient of log-likelihood) requires symmetric weights. The DNG uses directed edges with no symmetry enforcement. In the non-convergent regime this doesn't prevent learning, but it makes the CHL signal noisier than necessary -- every update pushes in an *approximately* correct direction instead of a *maximally informative* one.

**Fix:** At template creation time, for every edge (j→i) also create (i→j). During CHL, average the correlation differences for paired edges before applying the update. This doesn't force identical weights -- it forces the *learning signal* to be symmetric, which is the part the theory requires.

**Cost:** Roughly doubles edge count. Mitigate by reducing fan-in slightly; the symmetric pairs carry more information per edge than unpaired directed edges.

**Theory:** Xie & Seung (2003) showed asymmetric CHL approximates the true gradient with error proportional to weight asymmetry. Symmetric pairs eliminate this error term entirely for paired edges.

---

## 2. Adaptive Phase Length (Correlation Stabilization)

**Problem:** Free and clamped phases are fixed at 40 steps. If the network hasn't settled, the recorded correlations reflect transient dynamics rather than the network's actual interpretation. This adds noise to CHL. For larger grids or deeper transformations, 40 steps may be systematically insufficient.

**Fix:** Run each phase until the correlation vector stabilizes: `max|corr(t) - corr(t-1)| / step < epsilon`. Cap at a generous maximum (200 steps) to prevent runaway. This makes the CHL signal quality *adaptive* to task complexity -- simple tasks settle fast, complex tasks get more compute.

**Cost:** Variable compute per task. In practice, simple tasks will terminate early (saving time), complex tasks will use more (but get a better signal). Net effect is better learning per CHL update.

**Theory:** Equilibrium correlations are the sufficient statistics for CHL. Closer to equilibrium → lower variance gradient estimate → more efficient learning. This is analogous to increasing the sample count in stochastic gradient methods.

---

## 3. Signed Internal Activations

**Problem:** The code uses rectified rates `r = clip(V - threshold, 0, max_rate)` for all nodes. With non-negative rates, pre\*post correlations are always ≥ 0. The CHL difference still works (positive means "more active with answer present") but loses **anti-correlation information**: two nodes that are mutually suppressive produce correlation 0, same as two inactive nodes.

**Fix:** Use tanh activations for internal and concept nodes (as the math doc already specifies). Keep rectified rates for sensory and motor, where non-negativity aligns with one-hot encoding. Record correlations on the signed activations for internal-involving edges.

**Benefit for continuous learning:** Anti-correlations let inhibitory pathways be trained explicitly. Currently, inhibitory nodes participate in WTA competition but their specific connections aren't shaped by CHL as effectively -- they can only have "less positive" correlations, not negative ones. Signed activations give CHL a richer signal to sculpt the E/I balance.

**Theory:** Information-theoretically, signed activations carry 1 bit more per correlation measurement (the sign). Over thousands of edges and hundreds of CHL updates, this compounds into substantially more efficient learning.

---

## 4. Scale the Abstract Pool with Task Count

**Problem:** With 480 abstract neurons and 5% WTA, each task claims ~24 neurons. Maximum non-overlapping capacity is ~20 tasks. Real ARC requires hundreds.

**Fix:** Make the abstract pool size proportional to the expected task repertoire. For N tasks, allocate at least `N * k / WTA_frac` abstract neurons, where k is an overlap tolerance factor (k ≈ 1.5 gives comfortable margin). For 100 tasks at 5% WTA: `100 * 1.5 / 0.05 = 3000` abstract neurons.

**Alternative:** Hierarchical WTA -- multiple pools at different abstraction levels, each with its own sparsity. Lower pools handle perceptual features (shared across many tasks), upper pools handle task-specific configurations. This is capacity-efficient because shared features don't count against per-task budgets.

**Theory:** Kanerva's Sparse Distributed Memory gives the capacity bound: for k-of-n codes with random patterns, capacity scales as `n / (k * log(n))`. At n=3000, k=150 (5%), capacity ≈ 3000 / (150 * 11.6) ≈ 1.7 -- wait, that's pattern *retrieval* capacity for autoassociative memory. For the DNG, tasks use *learned* (not random) codes, so overlap is structured and capacity is higher. Empirical calibration needed, but the direction is clear: more neurons → more tasks.

---

## 5. Consolidation Decay (Prevent Ossification)

**Problem:** Consolidation only increases: `consolidation += strength` after success. Over long continuous learning, early-learned tasks accumulate very high consolidation, making those synapses essentially frozen. If the environment changes (task distribution shifts), the network can't unlearn.

**Fix:** Add slow consolidation decay: `consolidation *= (1 - decay)` each sleep cycle, with `decay` very small (0.01-0.02). Synapses that aren't periodically reinforced by replay gradually become plastic again. Synapses that are replayed nightly maintain their consolidation via the reward-triggered increment.

**Effect:** Creates a "use it or lose it" dynamic for stability. Tasks that remain in the replay buffer stay consolidated. Tasks that drop out of the buffer (e.g., buffer overflow, relevance fading) gradually become overwritable. This turns consolidation from a monotonic ratchet into a dynamic equilibrium.

**Theory:** This mirrors biological synaptic tagging and capture -- tags expire unless reinforced by protein synthesis (replay). Fusi et al. (2005) showed that cascade models with multiple plasticity timescales and decay at each level achieve optimal memory lifetime scaling.

---

## 6. Lateral Inhibition Between Task Representations

**Problem:** WTA within the abstract pool is global -- all abstract neurons compete in one flat pool. This means two tasks that happen to activate nearby neurons interfere at the WTA level even if their weights are fully consolidated.

**Fix:** Cluster the abstract pool into sub-pools (groups of ~50-100 neurons), each with its own local WTA. Tasks naturally segregate into different sub-pools. Cross-pool inhibition is weaker than within-pool, so multiple sub-pools can be active simultaneously -- one per active "concept."

**Benefit for compositionality:** If task A uses sub-pool 1 and task B uses sub-pool 3, a composed task "A then B" could activate both sub-pools simultaneously, with each contributing its learned transformation. Flat global WTA prevents this because only one sparse pattern can win.

**Theory:** This is cortical column theory (Mountcastle, 1997). Each column processes one feature independently. The binding problem is solved by temporal synchrony or spatial routing, not by cramming everything into one competitive pool.

---

## 7. Wire In Episodic Memory for Fast Task Binding

**Problem:** `EpisodicMemory` is implemented but not used in `train_full.py`. The network must learn task identity purely from CHL weight changes. For ARC's few-shot setting, this is slow -- CHL needs multiple rounds to encode a new task, and the encoding is permanent (weights change).

**Fix:** Enable episodic memory as a **fast complementary system**. On first exposure, store input-output pairs verbatim. On test, recall the most similar stored pair and inject it as a motor hint. This gives the network an immediate "first guess" without any weight change. CHL then refines the guess by adjusting weights to reproduce the recalled pattern more reliably.

**Benefit for continuous learning:** Episodic memory provides instant task performance (1-shot), while CHL provides gradual consolidation into the weight structure (slow learning). Tasks that are encountered frequently get consolidated; rare tasks rely on episodic recall. This is exactly the Complementary Learning Systems architecture (McClelland et al., 1995).

**Theory:** Biologically, hippocampal replay (your sleep mechanism) transfers episodic memories into neocortical weights. The loop is: fast episodic encoding → sleep replay → slow CHL consolidation → episodic memories can be pruned. You have all the pieces; they just aren't connected.

---

## 8. CHL on Temporal Sequences (Not Just Static Correlations)

**Problem:** Current CHL records average correlations over all timesteps in a phase. This discards temporal structure -- the network's dynamics form a trajectory, and the order of activations carries information about the transformation being computed. Averaging collapses this.

**Fix:** Record correlations at multiple time lags: `corr_tau[k] = mean(r_src(t) * r_dst(t + tau))` for tau in {0, 1, 2, 5}. Use the full set of lagged correlations in the CHL update. Edges that need to carry sequential information (A activates, then B activates 2 steps later) get a learning signal from `tau=2` that the `tau=0` correlation misses.

**Benefit:** Spatial transformations like flip require sequential propagation: sensory → internal → motor. The correct correlations appear at a time lag, not simultaneously. Lagged CHL captures this causal structure directly.

**Theory:** This is the temporal difference between STDP (spike-timing-dependent plasticity) and rate-based Hebbian learning. STDP's sensitivity to temporal order is what makes it powerful for learning causal/sequential structure. Lagged correlations approximate this in the rate-coding regime.

---

## Priority Ranking

| # | Improvement | Effort | Expected Impact | Risk |
|---|------------|--------|----------------|------|
| 5 | Consolidation decay | Trivial (3 lines) | High -- prevents long-term ossification | Low |
| 7 | Wire in episodic memory | Low (already built) | High -- instant few-shot, proper CLS loop | Low |
| 2 | Adaptive phase length | Low | Medium -- better CHL signal quality | Low |
| 4 | Scale abstract pool | Low (config change) | High -- needed for >20 tasks | Low |
| 3 | Signed internal activations | Medium (kernel changes) | Medium -- richer CHL gradients | Medium (may need retuning) |
| 1 | Symmetric CHL pairs | Medium (template + plasticity) | Medium -- cleaner learning signal | Medium (more edges) |
| 6 | Hierarchical WTA | High (architecture change) | High -- compositionality | High (complex interactions) |
| 8 | Lagged correlations | High (kernel rewrite) | Uncertain -- theoretically motivated but untested | Medium |
