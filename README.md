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



# Speed test

## n vs time

(b = 8, d = 64).

Forward mode:

![](./image/n_GPU_forward.jpg)

Backward mode:

![](./image/n_GPU_backward.jpg)



## d vs time

(b = 8, n = 2048).

Forward mode:

![](./image/d_GPU_forward.jpg)

Backward mode:

![](./image/d_GPU_backward.jpg)



## b vs time

(d = 512, n = 2048).

Forward mode:

![](./image/b_GPU_forward.jpg)

Backward mode:

![](./image/b_GPU_backward.jpg)

# Training speed compared to transormer

```
## fp32
transformer:
small: "wps": "5426.5", "ups": "1.32", "wpb": "4096
medium: "wps": "2630.2", "ups": "0.64", "wpb": "4096"

tnn(cuda)
small: "wps": "6100.3", "ups": "1.49", "wpb": "4096"
medium: "wps": "3081.4", "ups": "0.75", "wpb": "4096"

tnn(naive)
small: "wps": "5354.6", "ups": "1.31", "wpb": "4096"
medium: "wps": "2687.5", "ups": "0.66", "wpb": "4096"

## fp16
transformer:
small: "wps": "30159.9", "ups": "7.36", "wpb": "4096"
medium: "wps": "17607", "ups": "4.3", "wpb": "4096"

tnn(cuda)
small: "wps": "21793.7", "ups": "5.32", "wpb": "4096"
medium: "wps": "12290.8", "ups": "3", "wpb": "4096"

tnn(naive)
small: "wps": "13463", "ups": "3.29", "wpb": "4096"
medium: "wps": "7453.3", "ups": "1.82", "wpb": "4096"
```

# Todo


- [x] Implemention use in our paper.
- [x] Add note_cnt.
- [x] Add forward and backward derivation.
- [x] Add forward and backward code(causal v1).
- [x] Add forward and backward value check(causal).
- [x] Add speed test(causal).
- [x] Add forward and backward code(causal v2).
- [x] Add no transpose version(causal v3).
- [x] Add cpu/gpu speed stat and curve.
- [x] Add develop log.
- [x] Add profile.
- [x] Porting fftconv from H3.
- [x] Add Block FFT.
- [x] Fix fftconv value bug.
- [ ] Add memory stat and curve.
- [ ] Add support for fp16 and bf16.



# Reference

- [https://gist.github.com/iacolippo/9611c6d9c7dfc469314baeb5a69e7e1b](https://gist.github.com/iacolippo/9611c6d9c7dfc469314baeb5a69e7e1b)
- [https://github.com/BlinkDL/RWKV-CUDA](https://github.com/BlinkDL/RWKV-CUDA)
- [https://github.com/HazyResearch/H3](https://github.com/HazyResearch/H3)