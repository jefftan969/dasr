# Distilling Neural Fields for Real-Time Articulated Shape Reconstruction
Jeff Tan, Gengshan Yang, and Deva Ramanan (CVPR 2023)

### Goal: Train feed-forward shape and motion predictors by distilling differentiable rendering optimizers (e.g. category-level dynamic NeRFs)

### Links
- ([Project Page](https://jefftan969.github.io/dasr)) ([Paper](https://jefftan969.github.io/dasr/paper.pdf)) ([Poster](https://jefftan969.github.io/dasr/poster.pdf)) ([Video](https://www.youtube.com/watch?v=taUtXtW8b3Q))

### Key Features
- Representation
  - Rest shape: Category-level meshes
  - Deformation: Skeleton + linear blend skinning
  - Appearance: Global appearance embeddings
- Architecture
  - Category-level priors: Derived from dynamic NeRF teacher
  - Temporal encoder: Smooth latent codes over time, avoid jitter
  - Camera multiplex: Improve optimization when many poses are likely

### Timeline
- [ ] Evaluation code: mid June
- [ ] Full release (dataset, training code, demo): early July
