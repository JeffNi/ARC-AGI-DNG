# Biological Inefficiencies & Improvement Ideas

Mechanisms we follow from biology that may be wasteful at our scale (~4000 neurons vs 86 billion), with ideas for doing better if we get a working system.

---

## 1. Developmental Cell Death (Apoptosis)

**Biology:** ~50% of cortical neurons die during development. Neurons compete for limited neurotrophic factors (BDNF, NGF); losers undergo programmed cell death. Well-documented: Oppenheim (1991) "Cell death during development of the nervous system", *Annual Review of Neuroscience*; Southwell et al. (2012) showed transplanted interneurons follow the same death schedule regardless of host age.

**Why it's wasteful for us:** With only 1500 internal neurons, losing 50% means 750 neurons doing the work. We don't have 86 billion to spare. Every dead neuron is a significant fraction of our computational capacity.

**Possible improvements:**
- Instead of overproducing neurons and culling, could we start with fewer neurons and grow on demand? Synaptogenesis already grows connections — could we add neurogenesis for genuinely new processing units?
- Alternatively, recycle dead neurons: reset their weights and excitability, give them new random connections, and let them compete again. This is biologically implausible but computationally efficient.
- Use a functional homeostasis (see §5) to keep most neurons productive without fighting competitive learning.

**Status:** Currently we let neurons die naturally via health-based pruning. We don't add new neurons, only new connections. This is a known limitation.

---

## 2. Synaptic Scaling Fights Competitive Learning

**Biology:** Turrigiano (1998, 2008) showed neurons multiplicatively scale all incoming weights to maintain a target firing rate. This maintains stability in adult circuits.

**Problem during development:** Scaling pushes all neurons toward the same target rate, directly opposing competitive differentiation. Winners get suppressed, losers get boosted. In adult cortex this preserves learned structure; during the critical period it prevents structure from forming.

**Current fix:** Disabled scaling and intrinsic plasticity during infancy. Stability provided by L2 weight normalization, conscience mechanism, SHY downscaling, and health-based pruning instead.

**Possible improvements:**
- Develop a "functional homeostasis" that optimizes for representational diversity rather than uniform firing rate. Target metric: maximize information content of the neural population, not equalize individual rates.
- Asymmetric scaling: only suppress pathologically over-active neurons (anti-seizure), don't boost quiet ones during development.
- Rate-conditional scaling: only apply to neurons above a minimum activity level (active but mistuned, not dead).

---

## 3. Slow Synaptogenesis

**Biology:** New synapse formation requires physical growth of dendritic spines and axonal boutons. Takes hours to days per synapse. At our scale, reaching the 4x growth target from birth takes unreasonably long.

**Problem for us:** With 50,000 candidates per step and ~0.3 acceptance rate, growth is still only ~1.04x per day. Reaching 4x growth takes months of simulated time.

**Possible improvements:**
- Batch synaptogenesis: grow connections in larger bursts during sleep rather than trickle during waking.
- Adaptive growth rate: grow faster when representational diversity is low (lots of room for new features), slower when it's high (network is mature).
- Targeted growth: instead of random co-active pairs, preferentially connect neurons that would increase population diversity.

---

## 4. Uniform Initial Weights

**Biology:** Synapses form with varied initial strengths depending on molecular signals. We use uniform weights for simplicity.

**Problem:** All neurons start identical, so competitive learning must build ALL differentiation from scratch using only input statistics. This is slow and fragile.

**Possible improvements:**
- Initialize weight vectors with small random perturbations so neurons start slightly different. Even 5% variation would break the initial symmetry and give competitive learning a head start.
- Use structured initialization based on topographic position (neurons near different grid regions get slightly different initial biases).

**Status:** We added position encoding to features (retinotopic), which helps. Could also add position-dependent weight initialization.

---

## 5. Functional Homeostasis (Speculative)

**Concept:** Instead of mimicking biology's homeostasis (which evolved for 86B neurons and can afford waste), design a homeostasis that optimizes for our actual goals:

- **Population diversity target:** Instead of "every neuron fires at rate X", target "the population should have high representational diversity (low sim)."
- **Utility-based survival:** Instead of "neurons below target rate get boosted", use "neurons that contribute unique information survive; redundant neurons get recycled."
- **Adaptive competition:** WTA fraction adjusts based on current sim/selectivity, not just a developmental clock.

This deviates from biology but may be necessary at our scale. File under "if we get the basics working first."

---

## 6. Sleep Architecture

**Biology:** Neonatal sleep is very different from adult sleep — more REM, different oscillation patterns, different consolidation mechanisms. We apply adult-like SHY downscaling uniformly.

**Problem:** We don't know if neonatal SHY operates the same way. Our sleep module may be inappropriate for infancy.

**Possible improvements:**
- Research neonatal sleep consolidation mechanisms specifically.
- Consider whether sleep during infancy should focus on synapse GROWTH (matching neonatal biology) rather than downscaling.

---

*This doc tracks ideas. Implement only after core competitive learning is working.*
