---
layout: page
title: Generative molecular AI
description: A fun side project
category: fun
---

I recently started working on a [generative model for molecule design](https://github.com/jphaupt/generative-molecular-design).

Admittedly, I underestimated the complexity of this endeavour. My eventual goal is
to train an equivariant message-passing graph variational autoencoder with variable input
and output, written in PyTorch, and use a basic reinforcement learning algorithm
to search for a molecule similar to those in QM9 with the highest HOMO-LUMO gap
(as a surrogate to more interesting and complex properties). Additional molecules
generated would have their targets calculated with PySCF.

I have some functions done for the final model (e.g. equivariant layers, PySCF helpers)
but at the moment the model only does basic predictions, and takes padded fixed-size inputs.
I hope to revisit this project more earnestly once I have released PyTCHInt. :)
