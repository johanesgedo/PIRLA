# PIRLA

**Physics-Informed Reinforcement Learning Agent**
[![DOI]()]()

PIRLA is and experimental framework that combines reinforcement learning with physics-informed learning in a unified engine.

This repository is intended for research and experimental on artificial intelligence systems governed by physical constraints, hierarchical temporal structure, and adaptive learning mechanisms, designed to solve inverse problems and seismic imaging.

> **DOI:** 

---

## Overview

PIRLA combines several major paradigms:

1. **Reinforcement Learning (RL)** for reward-based decision making
2. **Physics-Informed Learning** for incorporating physical constraints, PDE residuals, or domain structure into optimization
3. **Adaptive Meta-Learning** for dynamically adjusting temperature, exploration, and epsilon.
4. **Hierarchical Temporal Abstraction** for organizing actions at the option level, feudal policy level, or hybrid schemes.
5. **Experience Replay and Policy Optimization** for storing transitions, sampling experience, and computing policy loss and advantage.
6. **Modular Scientific Engine** for easier integration with geophysical software and data.

---

## Key Features
- Adaptive action selection based on calculation conditions;
- Automatically adjustable global and local temperatures;
- Exploration control that depends on reward variance and entropy, not on a single fixed rule;
- Proximal Policy Optimization style clipped policy loss;
- Kullback-Leibler regularization for stable policy updates;
- Extensible experience replay buffer;
- Hierarchical RL through options and feudal control;
- Integration of PINNs for incorporating physical knowledge;
- CuPy-based implementation for GPU acceleration where available.

---

## Possible Extensions

The framework can be extended for:

- Adaptive control optimization;
- Physics-driven geophysical processing;
- Parameter inference for PDE systems;
- Hybrid learning for scientific simulation.

---

## Project Status

PIRLA is a research and prototype framework currently under development.
It is currently best suited for:

- Concept validation;
- Geophysical experiments;
- Solving inverse problem and seismic imaging

---

## License

PIRLA is distributed under the terms and conditions specified in the `LICENSE` file included in this repository.

An archived release of PIRLA is publicly available through Zenodo and can be cited using the following DOI:

> **Citation:** 






