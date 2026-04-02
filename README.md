# Dynamic Neural Graph (DNG)

> A research project exploring whether intelligence can emerge from structure and plasticity rather than scale and gradient descent.

---
# This is a work in progress

## Overview

Modern LLMs have more parameters than the human brain has synapses — yet they can't solve puzzles a child handles effortlessly. The problem isn't scale. It's architecture.

LLMs are feedforward pattern matchers trained via backpropagation on massive corpora. They have no persistent state, no recurrence, no way to *think* through a novel problem. Every capability must be baked in at training time. Despite trillion-parameter budgets, they fail at the kind of flexible, abstract reasoning that biological brains do "cheaply" with ~86 billion neurons.

This project asks: **what if we went back to recurrence?**

## What Is This?

A **Dynamic Neural Graph (DNG)** — a recurrent network whose connectivity mutates at runtime in response to flowing signals. Instead of converging through gradient descent on a fixed architecture, the DNG:

- Starts from a **genetically architected seed network** encoding innate computational primitives
- Contains **all nodes from initialization** (no runtime neurogenesis — matching biology)
- Begins with **overproduced connections** sculpted via activity-dependent pruning
- Uses **local, biologically-plausible learning rules** (no backpropagation)
- Forms stable activation patterns (**attractors**) that represent understanding
- **Learns continuously** — new knowledge integrates without destroying old knowledge

The target benchmark is [ARC-AGI](https://arcprize.org/), a set of abstract reasoning tasks that humans solve trivially but AI largely cannot.

## Core Hypothesis

**Intelligence emerges from the interaction of signal dynamics and structural plasticity in a pre-structured recurrent network.**

The brain doesn't do backpropagation. It doesn't have a loss function. But it also doesn't start from random wiring — the genome specifies regional organization, topographic maps, neuron types, and circuit motifs *before any learning*. We replicate this: a structured genetic template provides the scaffold, and plasticity fills in task-specific details.

This is as much a cognitive science experiment as an engineering one. If a small recurrent system can exhibit flexible reasoning, that tells us something about what intelligence actually requires — and what it doesn't.

## Hardware

Designed for consumer hardware: ~20,000 nodes, seconds per ARC task on a laptop.

## Design Documents

All design documentation lives in `docs/`:

| Document | Contents |
|---|---|
| [Biological Foundations](docs/01_Biological_Foundations.md) | Neuroscience research informing the design — plasticity, cell assemblies, what's genetic vs. learned, perception-reasoning interaction |
| [Architecture](docs/02_Architecture.md) | Genetic template, functional regions, microcircuit motifs, node types, evolutionary search, scale |
| [Mathematics](docs/03_Mathematics.md) | Formal definitions, activation dynamics, plasticity rules, energy function, convergence |
| [ARC Strategy](docs/04_ARC_Strategy.md) | Signal encoding, task processing pipeline, what "thinking" looks like |
| [Related Work](docs/05_Related_Work.md) | Existing approaches (NEAT, Liquid NNs, NCA, program synthesis), LLM analysis, our differentiation |
| [Open Questions](docs/06_Open_Questions.md) | Resolved decisions, open design questions, known risks, roadmap |

## ARC-AGI State of the Art (private evaluation sets)

- **ARC-AGI-1:** Best Kaggle private ~53% (2024 competition). Public set scores (~98%) are inflated. Still unsolved at human level.
- **ARC-AGI-2:** Best Kaggle private ~24% (active competition, 2025). Far from solved.
- **ARC-AGI-3:** Interactive turn-based environments. Best AI 0.37%. Humans 100%.

We start with ARC-AGI-1 (same format as 2, simpler tasks, faster iteration).

## Roadmap

1. **Finalize design** — review and refine the theoretical framework
2. **Pencil-and-paper example** — trace dynamics through a small network on a trivial task
3. **Minimal prototype** — Python implementation, test on simple ARC tasks
4. **Evolutionary template search** — find good seed network architectures
5. **Iterate and scale** — refine plasticity rules, evaluate on full ARC-AGI
