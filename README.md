# fast-tno

This repository aims to optimize the Tno operator proposed in the paper [**Toeplitz Neural Network for Sequence Modeling**](https://openreview.net/pdf?id=IxmWsm4xrua). The main equation is the following format:


$$
\mathrm{Tno}: \mathbf x \to \mathbf o,\\
\mathbf o= \mathbf T \mathbf x , \mathbf T\in \mathbb R^{n\times n}, \mathbf x, \mathbf o \in \mathbb R^{n\times 1}.
$$

In practice, we use the Tno operator in each feature dimension, so the complete formula is as follows:

$$
\mathbf O[:, i]= \mathbf T_i \mathbf X[:, i], \\
\mathbf O[:, i]\in \mathbb R^{n\times 1}, \mathbf T_i\in \mathbb R^{n\times n},  \mathbf X[:, i]\in \mathbb R^{n\times 1}.
$$

Although the theoretical complexity is $O(nd\log n )$, it is slower than Attention when $n$ is small, so there is still a lot of room for optimization.

# ToDo


- [x] Implemention use in our paper.
- [x] Add note_cnt.
- [x] Add forward and backward derivation.
- [x] Add forward and backward code(causal v1).
- [x] Add forward and backward value check(causal).
- [x] Add speed test(causal).
- [x] Add forward and backward code(causal v2).
- [x] Add no transpose version(causal v3).
- [x] Add stat and curve.
- [ ] Add support for fp16 and bf16.
- [ ] Add develop log.



# Reference

- [https://gist.github.com/iacolippo/9611c6d9c7dfc469314baeb5a69e7e1b](https://gist.github.com/iacolippo/9611c6d9c7dfc469314baeb5a69e7e1b)
- [https://github.com/BlinkDL/RWKV-CUDA](https://github.com/BlinkDL/RWKV-CUDA)
