# 中文笔记

## 前向和反向

首先给出Tno的核心操作：

$$
\begin{aligned}
\mathbf o&= \mathbf T \mathbf x\\
\mathbf{T}&=\left[\begin{array}{cccc}
t_0 & t_{-1} & \cdots & t_{-n+1} \\
t_1 & t_0 & & \vdots \\
\vdots & & t_0 & t_{-1} \\
t_{n-1} & \cdots & t_1 & t_0
\end{array}\right] \in \mathbb{R}^{n \times n}
 \\
\mathbf T&\in \mathbb R^{n\times n}, \mathbf x, \mathbf o \in \mathbb R^{n\times 1}
\end{aligned}
$$

写成求和形式为：

$$
o_i= \mathbf \sum_{j=0}^{n-1}t_{i-j} x_j
$$

在language model中，因为单向的原因，上三角部分都为0，即：

$$
\begin{aligned}
\mathbf o&= \mathbf T \mathbf x\\
\mathbf{T}&=\left[\begin{array}{cccc}
t_0 & 0 & \cdots &0 \\
t_1 & t_0 & & \vdots \\
\vdots & & t_0 & 0 \\
t_{n-1} & \cdots & t_1 & t_0
\end{array}\right] \in \mathbb{R}^{n \times n}
 \\
\mathbf T&\in \mathbb R^{n\times n}, \mathbf x, \mathbf o \in \mathbb R^{n\times 1}
\end{aligned}
$$

写成求和形式为：

$$
o_i= \mathbf \sum_{j=0}^{i}t_{i-j} x_j
$$

注意为了高效实现，两者对应的cuda应该会有所不同，为了区分这两种情形，将一般情形成为non causal，将language model的单向情形成为causal，后续会分别进行讨论。

对于多维版本，主要逐维度操作即可。



### non causal

#### 前向

输入：

- $\mathbf t= [t_{n-1}, \ldots, t_0, \ldots, t_{1-n}]^\top \in \mathbb R^{(2n-1)\times 1}$；
- $\mathbf x \in \mathbb R^{n\times 1}$；

输出：

- $\mathbf o= \mathbf T \mathbf x \in \mathbb R^{n\times 1}$；
- $o_i= \mathbf \sum_{j=0}^{n-1}t_{i-j} x_j$；



#### 反向

注意到：

$$
 o_i= \mathbf \sum_{j=0}^{n-1}t_{i-j} x_j=\mathbf \sum_{j=i-(n-1)}^{i}x_{i-j} t_j
$$

所以：

$$
\frac{\partial o_i}{\partial t_j}= x_{i-j}, j=i-(n-1), \ldots, i
$$

写成矩阵形式为：

$$
\mathbf a\triangleq \frac{\partial \mathbf o}{\partial \mathbf t}
&=\left[\begin{matrix}
0 & \cdots &  0 & x_0 & x_1 & \cdots &x_{n-1} \\
0 & 0& x_0 & x_1 & \cdots &x_{n-1} & 0\\
\vdots & \vdots & \vdots  & \vdots & \vdots  &\vdots  & \vdots  \\
x_0 & x_1 & \cdots &x_{n-1} & 0 & 0 & 0
\end{matrix}\right] \in \mathbb R^{n\times (2n- 1)}
$$

假设：

$$
\mathbf b \triangleq \nabla_{\mathcal L} \mathbf o \in \mathbb R^{n\times 1}
$$

那么：

$$
\begin{aligned}
\mathbf c &\triangleq  \left[\frac{\partial \mathbf o}{\partial \mathbf t}\right]^\top  \nabla_{\mathcal L} \mathbf o  \in \mathbb R^{(2n-1)\times 1}\\
&= \left[\begin{matrix}
x_{n-1} & 0 &  \cdots& 0  \\
x_{n-2}  & x_{n-1} &0 & x_1 \\
\vdots & & \vdots  & \vdots &   \\
x_0 &x_1  & \cdots & x_{n-1} \\
0 & x_0 & \cdots &x_{n-1} \\
\vdots & \vdots & \vdots & \vdots \\
0& 0 & 0 & x_0 
\end{matrix}\right]\mathbf b   \\
c_i & = \begin{cases}
\sum_{j=0}^{i} x_{n-1-j}b_j & i \le n - 1\\
\sum_{j=0}^{2n-2-i} x_{j}b_{j+i-n+1} & i\ge n
\end{cases}
\end{aligned}
$$

小结一下：

输入：

- $\mathbf b \triangleq \nabla_{\mathcal L} \mathbf o \in \mathbb R^{n\times 1}$；

输出：

- $\mathbf c \triangleq  \left[\frac{\partial \mathbf o}{\partial \mathbf t}\right]^\top  \nabla_{\mathcal L} \mathbf o  \in \mathbb R^{(2n-1)\times 1}$；
- $c_i  = \begin{cases}
  \sum_{j=0}^{i} x_{n-1-j}b_j & i \le n - 1\\
  \sum_{j=0}^{2n-2-i} x_{j}b_{j+i-n+1} & i\ge n
  \end{cases}$；



### causal

#### 前向

输入：

- $\mathbf t= [t_0, \ldots, t_{n-1}]^\top \in \mathbb R^{n\times 1}$；
- $\mathbf x \in \mathbb R^{n\times 1}$；

输出：

- $\mathbf o= \mathbf T \mathbf x \in \mathbb R^{n\times 1}$；
- $o_i= \mathbf \sum_{j=0}^{i}t_{i-j} x_j$；



#### 反向

注意到：

$$
o_i= \mathbf \sum_{j=0}^{i}t_{i-j} x_j=\mathbf \sum_{j=0}^{i}x_{i-j} t_j
$$

所以：

$$
\frac{\partial o_i}{\partial t_j}= x_{i-j},\frac{\partial o_i}{\partial x_j}= t_{i-j}, j=0, \ldots, i 
$$

写成矩阵形式为：

$$
\begin{aligned}
\mathbf a\triangleq \frac{\partial \mathbf o}{\partial \mathbf t}&=\left[\begin{matrix}
x_0  & 0 & \cdots &  0 \\
x_1 & x_0 &  \cdots & 0 \\
\vdots & \vdots& \vdots  & \vdots   \\
x_{n-1}& x_{n-2} & \cdots &x_{0} 
\end{matrix}\right]\in \mathbb R^{n\times n} \\
\mathbf b\triangleq \frac{\partial \mathbf o}{\partial \mathbf x}&=\left[\begin{matrix}
t_0  & 0 & \cdots &  0 \\
t_1 & t_0 &  \cdots & 0 \\
\vdots & \vdots& \vdots  & \vdots   \\
t_{n-1}& t_{n-2} & \cdots &t_{0} 
\end{matrix}\right]\in \mathbb R^{n\times n}
\end{aligned}
$$

假设：

$$
\mathbf c \triangleq \nabla_{\mathcal L} \mathbf o \in \mathbb R^{n\times 1}
$$

那么：

$$
\begin{aligned}
\mathbf d
&= \nabla_{\mathcal L} \mathbf t \\
&\triangleq  \left[\frac{\partial \mathbf o}{\partial \mathbf t}\right]^\top  \nabla_{\mathcal L} \mathbf o  \in \mathbb R^{n\times 1}\\
&= \left[\begin{matrix}
x_0 &x_1  & \cdots & x_{n-1} \\
0 & x_0 & \cdots &x_{n-2} \\
\vdots & \vdots & \vdots & \vdots \\
0& 0 & 0 & x_0 
\end{matrix}\right]\mathbf c  \\
d_i & = 
\sum_{j=0}^{n - 1 -i} x_{j}c_{i+j} \\
\mathbf e
&= \nabla_{\mathcal L} \mathbf x \\
&\triangleq  \left[\frac{\partial \mathbf o}{\partial \mathbf x}\right]^\top  \nabla_{\mathcal L} \mathbf o  \in \mathbb R^{n\times 1}\\
&= \left[\begin{matrix}
t_0 &t_1  & \cdots & t_{n-1} \\
0 & t_0 & \cdots &t_{n-2} \\
\vdots & \vdots & \vdots & \vdots \\
0& 0 & 0 & x_0 
\end{matrix}\right]\mathbf c  \\
e_i & = 
\sum_{j=0}^{n - 1 -i} t_{j}c_{i+j}
\end{aligned}
$$

小结一下：

输入：

- $\mathbf b \triangleq \nabla_{\mathcal L} \mathbf o \in \mathbb R^{n\times 1}$；
- $\mathbf c \triangleq \nabla_{\mathcal L} \mathbf o \in \mathbb R^{n\times 1}$；

输出：

- $\mathbf d \triangleq  \left[\frac{\partial \mathbf o}{\partial \mathbf t}\right]^\top  \nabla_{\mathcal L} \mathbf o  \in \mathbb R^{n\times 1}$；
- $d_i  = \sum_{j=0}^{n - 1 -i} x_{j}c_{i+j}$​；
- $\mathbf e \triangleq  \left[\frac{\partial \mathbf o}{\partial \mathbf x}\right]^\top  \nabla_{\mathcal L} \mathbf o  \in \mathbb R^{n\times 1}$；
- $e_i  = \sum_{j=0}^{n - 1 -i}t_{j}c_{i+j}$​；





### 小结

从上述推导可以看出，需要实现三个cuda，分别是下三角，上三角以及一般情形的Toeplitz matrix vector production。





## CUDA版本

### V1

- 维度：$\mathbf T: (d, n), \mathbf x:(b, d,n)$；
- $b, d$维度分thread；



### V2

- 维度：$\mathbf T: (d, n), \mathbf x:(b, d,n)$；
- $b, d$维度分thread，$n$维度分thread；



### V3

- 维度：$\mathbf T: (n, d), \mathbf x:(b, n, d)$；
- $b, d$维度分thread，$n$维度分thread；



### V4

- 维度：$\mathbf T: (n, d), \mathbf x:(b, n, d)$；
- 在$b, d, n$维度并行处理；
- $b, d, n$分别分thread；



### V5

- 维度：$\mathbf T: (n, d), \mathbf x:(b, n, d)$；
- 在$b, d, n$维度并行处理；
- $b, d, n$分别分thread；
- 使用shared memory；
  - forward mode；



### V6

- 维度：$\mathbf T: (n, d), \mathbf x:(b, n, d)$；
- 在d, n$维度并行处理；
- $d, n$分别分thread；
