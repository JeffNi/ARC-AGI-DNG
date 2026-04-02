# ARC-AGI Strategy

How the DNG is applied to solve ARC tasks.

---

## 1. Core Principle: Raise a Child

The DNG is not "trained" -- it is **raised**. Like a child who is born with a brain structure (the genome) and develops reasoning through years of interactive experience.

The network doesn't passively absorb data. It:
- **Sees** a problem (sensory input)
- **Tries** to answer it (motor output -- the "mouth")
- **Sees** the correct answer (sensory feedback)
- **Feels** reward if it got it right (dopamine signal)
- **Rests** and **sleeps** to consolidate what it learned

Over many "days" of this cycle, the network develops general reasoning circuits through the overshoot-prune-stabilize process. When mature, it can solve novel problems it has never seen.

---

## 2. Signal Encoding

An ARC grid (H x W, values 0-9):

- **One-hot encoding:** One node per cell per color = H x W x 10 nodes. Color c -> c-th node at +1, all others at 0.
- **Sensory window:** Fixed-size, one grid at a time. Grids are presented sequentially.

---

## 3. A Day in the Life

Each "day" is one development cycle. The network wakes up, studies, rests, and sleeps.

### Study session

The network works through several ARC tasks using an **observation-first** paradigm:

```
for each task today:

    1. OBSERVE ALL EXAMPLES (passive, no testing)
       For each training pair (input, output):
         - Clamp both input AND output into the sensory/motor window
         - Recurrent dynamics run -- the network absorbs the pattern
         - ACh is high (encoding mode)
         - No weight updates -- just building dynamic state
         - Facilitation and LTM nodes accumulate context

    2. ATTEMPT THE TEST (active, with learning)
       For each attempt:
         a. FREE PHASE: Clamp only the test input. Motor neurons are free.
            The network thinks using its dynamic state (facilitation,
            memory from observing examples) and produces a guess.
            Record correlations across all synapses.

         b. CLAMPED PHASE: Clamp both test input AND correct output.
            Record correlations again. The network "sees" the right answer.

         c. GENTLE CHL UPDATE: ΔW = eta * 0.1 * (clamped - free).
            Tunes the scaffold; doesn't memorize the specific task.

         d. NEUROMODULATOR UPDATE:
            - DA = reward (how close was the guess?)
            - NE = 1 - reward (urgency to explore)

         e. RE-OBSERVE examples to refresh context before next attempt.

    3. SYNAPTOGENESIS
       New connections grow between co-active neurons.
       Rate modulated by ACh (high in childhood).
```

### Rest / Play

After the study session, the network runs with no external input. Internal dynamics replay recent activity. Noise enables exploration. Plasticity continues during rest -- this is when insights crystallize.

### Sleep

Global weight downscaling + pruning + homeostatic regulation. Weak connections die. Strong ones survive. The network wakes up tomorrow cleaner and more focused.

---

## 4. Two Levels of Learning

### Level 1: Childhood (structural development)

The network is exposed to many diverse ARC tasks over many "days." Through interactive learning, its connections permanently change:
- Overproduced connections are pruned to useful circuits
- Spatial awareness, pattern detection, and rule abstraction emerge
- The motor region learns when and what to output
- The reward signal shapes which circuits get reinforced

This is **permanent structural learning** -- the overshoot-prune-stabilize cycle.

### Level 2: Task-solving (thinking)

A mature network sees a novel ARC task. It doesn't rewire itself -- it uses its developed circuits to think:
- Observes the examples through its sensory window
- Recurrent dynamics process the information using existing circuits
- Working memory (low-leak memory nodes) holds information across viewings
- The motor region produces an answer when confident

This is **temporary activation patterns** -- thinking, not learning.

The distinction: childhood changes the hardware (connections). Task-solving uses the hardware (activations).

---

## 5. Neuromodulatory Signals

Three global scalar neuromodulators govern network state:

| Signal | Role | High | Low |
|---|---|---|---|
| **Dopamine (DA)** | Reward / punishment | "Good job" -- strengthens recent changes | Dip below baseline = "that was wrong" -- reverses recent changes |
| **Acetylcholine (ACh)** | Encoding / plasticity | Observation mode: absorb information | Performance mode: use what you've learned |
| **Norepinephrine (NE)** | Arousal / exploration | Stressed, exploring new strategies | Calm, exploiting known strategies |

These signals don't carry information about *what* to learn. They modulate *how much* and *in what way* the network changes. The actual content comes from Contrastive Hebbian Learning (CHL): the network compares its own guess (free phase) to the correct answer (clamped phase), and the difference drives weight updates. Neuromodulators scale and gate these updates.

---

## 6. Learning Rule: Contrastive Hebbian Learning

Instead of backpropagation (biologically implausible), the network learns via CHL:

1. **Free phase**: Input clamped, motor neurons free. The network produces a guess. Record correlations between all connected neurons.
2. **Clamped phase**: Both input AND correct output clamped. Record correlations again.
3. **Weight update**: `ΔW = eta * (clamped_corr - free_corr)`. Strengthens connections that match the correct answer, weakens those that produced the wrong one.

CHL reaches every synapse in the network (unlike backprop, which follows a computational graph). It is the closest known learning rule to biological Hebbian plasticity that can still solve the credit assignment problem.

The update is kept **gentle** (`eta * 0.1`) to tune the general "scaffold" rather than memorize individual tasks.

---

## 7. Mode Signaling: How the Network Knows What It's Doing

The network must distinguish between observing (learning) and performing (answering). Rather than explicit "mode bits," we use two naturally occurring signals:

**Signal 1: Neuromodulator context**
- **Observation mode**: High ACh, low NE → "encode this, stay calm"
- **Performance mode**: Lower ACh, higher NE → "apply what you know, be alert"
- **Frustrated rewatch**: High ACh + elevated NE → "pay closer attention this time"

**Signal 2: Motor neuron state**
- **Observation**: Motor neurons are externally clamped (receiving the answer signal). They fire stably and strongly. The network can "feel" this because external drive feels different from self-generated activity.
- **Performance**: Motor neurons are free. They fire based only on internal connections -- weaker, noisier, self-generated.

The network doesn't need to be told which mode it's in. The combination of neuromodulator levels and whether the motor neurons are being driven externally provides a naturally distinguishable internal state. Over development, the network learns to behave differently in each context.

---

## 8. Tutorial & Homework System

Beyond unstructured "play" with ARC tasks, the network receives **guided instruction** -- curated tutorials that break down reasoning concepts step by step, followed by homework to test transfer.

### Tutorials

Hand-curated JSON sequences that show a concept visually, frame by frame. Like a silent instructional video. Each tutorial lives in `tutorials/<task_category>/` and contains ordered frames showing:

1. The input pattern highlighted
2. The key feature or rule being demonstrated
3. The transformation being applied step by step
4. The final output

During tutorial observation:
- **ACh is high** (encoding mode)
- Motor neurons are **clamped** (the answer is shown, not generated)
- **No CHL weight updates** -- pure passive observation
- Facilitation and LTM nodes build representations of the demonstrated concept

### Homework

After a tutorial, the network receives a set of **related but different** ARC tasks that require the same concept. This tests transfer -- can it apply what it "saw" to new problems?

Weekly homework cycle:
```
Monday:     Watch tutorial (passive observation, high ACh)
Tue-Fri:    Attempt homework tasks (performance mode, CHL active)
Saturday:   Assessment -- measure improvement over the week
Sunday:     Sleep + reward/punishment based on weekly performance
```

### Re-watching

If the network is **stuck** (low reward, no improvement across multiple attempts), it is allowed to **re-watch the tutorial**. The key difference from the first viewing:

- The network's internal state carries **residual facilitation from failed attempts**
- High ACh (re-encoding) + elevated NE (frustrated attention)
- The same tutorial frames activate different patterns because the context has changed

This mirrors how re-reading a textbook after struggling with a problem is more informative than reading it cold. You know what confused you, so you attend to different details.

### Punishment Signal

If the network fails its homework after the full week:

- **DA dips below baseline** (negative reward signal). This actively weakens recently-strengthened synapses -- "the approach you've been developing is wrong."
- **NE spikes** -- forces high-exploration mode, breaking the failed strategy
- **Anti-Hebbian**: connections most active during wrong answers get selectively weakened

This is biologically grounded: dopamine dips below baseline are a well-documented punishment signal in the brain's reward system. They serve the opposite function of reward -- instead of reinforcing, they actively discourage.

### Spaced Repetition

Tasks the network has solved are scheduled for re-testing at increasing intervals (1 day, 3 days, 7 days, ...). This tests retention:
- If still solved → interval increases (consolidated)
- If forgotten → interval resets, task re-enters the active homework queue

---

## 9. The Lifecycle

```
GENOME (found by evolutionary search)
  |
  v
BIRTH (instantiate template -- overproduced connections)
  |
  v
CHILDHOOD (many "days" of study, rest, sleep)
  |  - Interactive learning on diverse ARC tasks
  |  - Connections form, strengthen, get pruned
  |  - General reasoning circuits emerge
  |
  v
MATURITY (save -- this IS the trained model)
  |
  v
NOVEL TASK (load mature network)
  |  - Observe examples through sensory window
  |  - Think using developed circuits
  |  - Motor region produces answer when confident
  |
  v
ANSWER
```

---

## 10. The Mouth (Motor Region)

The motor region is the network's output mechanism -- its "mouth." It is NOT an engineered readout layer. It is part of the network, subject to the same dynamics and plasticity.

**During childhood:** The motor region produces guesses. Early on, these are random garbage. Over time, the network learns:
- What patterns in the sensory input should produce what motor output
- When to activate the mouth (confidence calibration)
- When to stay silent (if unsure, don't guess)

**During novel tasks:** The motor region activates through learned internal connections. Integration-to-bound: the network "speaks" only when motor activations are confident and stable.

The network learns when to talk and when not to talk, just like a child.

---

## 11. Persistence

- **save()** serializes the full DNG state.
- **load()** restores it exactly.
- A mature network is saved after childhood and loaded for novel tasks.
- The saved model IS the developed brain -- connections, weights, everything.

---

## 12. The Role of the Genome

The genome parameterizes everything:
- Network structure (node counts, types, regions)
- Connectivity patterns and densities
- Growth rates (learning rate, sleep factor, prune timing)
- How the developmental trajectory unfolds

**Evolutionary search** tries many genomes. Each genome produces a network, which goes through childhood, and is evaluated on novel tasks. Genomes that produce good reasoners survive. This is artificial evolution finding the brain architecture that develops best.

Some genomes might produce "geniuses" -- networks that develop fast and generalize well. Others produce networks that never learn. The search finds the winners.

---

## 13. Why This Maps to Human Cognition

A human solving ARC:

1. Was born with a brain structure (genome)
2. Spent years learning about spatial patterns, objects, transformations (childhood)
3. Developed general reasoning through trial and error with feedback (interactive learning)
4. Looks at a novel ARC task and applies their developed reasoning (thinking)
5. Says the answer when they're confident (mouth + confidence)

We're doing the same thing, compressed. The genome is found by evolution. Childhood is accelerated on a computer. The mature network solves novel tasks using the circuits it developed.
