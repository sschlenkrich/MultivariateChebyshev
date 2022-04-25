---
title: "Efficient Implementation of Multivariate Chebyshev Interpolation Using High-level Linear Algebra Operations"
author: "Sebastian Schlenkrich"
date: "April 2022"
abstract: |
  In this paper...

linkReferences: true
---


# Multivariate Chebyshev Interpolation

Multivariate Chebyshev interpolation is a classical interpolation method. It is applied for various applications in mathematical finance as discussed e.g. in [@GGMM16] and [@Poe20]. An important implementation of Chebyshev polynomials is the *chebfun* MATLAB project, see [@DHT14].

In its classical tensor-based form multivariate Chebyshev interpolation suffers from the *curse of dimensionality*. This means that the computational effort grows exponentially with the number of input dimensions of the target function. There are various approaches to circumvent the exponential growth in computational effort. As an example, we mention tensor trains which are recently proposed for function approximation in [@AP21].

With this work we do not aim at lifting the curse of dimensionality. Instead, we want to show how general-purpose high-level linear algebra operations can be used to implement multivariate Chebyshev interpolation efficiently given its intrinsic constraints. The high-level linear algebra operations itself are typically implemented as efficiently as possible by delegating calculations to BLAS routines and by applying parallelisation. 

Chebyshev interpolation is specified on the $D$-dimensional cube $\left[-1, 1\right]^D$. We denote $p=\left(p_1,\ldots,p_D\right)$ the elements of that standardised domain $\left[-1, 1\right]^D$. Original model inputs $x$ are assumed to be defined on a general hyper-rectangular domain. Such general domain is transformed into the standardised domain via an element-wise affine transformation.

We use the notation $p\left(x\right)$ and $x\left(p\right)$ to describe the affine transformation from the general input domain to the standardised domain and vice versa.

For a scalar parameter $p_d\in \left[-1, 1\right]$ the Chebyshev polynomial of degree $j$ is denoted $T_j\left(p_d\right)$. The Chebyshev polynomial is defined via
$$
  T_j\left(p_d\right)
  = \cos\left( j \arccos\left(p_d\right)\right).
$$
An equivalent representation is given via the recursion
$$
\begin{aligned}
  T_0 \left(p_d\right) &= 1, \\
  T_1 \left(p_d\right) &= p_d, \\
  T_j \left(p_d\right) &= 2 p_d T_{j-1} \left(p_d\right) - T_{j-2} \left(p_d\right).
\end{aligned}
$$

Multivariate Chebyshev polynomials are defined as products of one-dimensional Chebyshev polynomials. Let $\bar j=\left(j_1, \ldots, j_D\right)$ be a multi-index and $p\in \left[-1, 1\right]^D$. The multivariate Chebyshev polynomial of degree $\bar j$ is
$$
  T_{\bar j} \left(p\right) = \prod_{d=1}^D T_{j_d} \left(p_d\right).
$$

Tensor based Chebyshev interpolation of a target function $y(x)$ for (multi-index) degree $\bar n = \left(n_1, \ldots, n_D \right)$ is given by
$$
  f(x)
  = \sum_{0\leq \bar j \leq \bar n} c_{\bar j} T_{\bar j} \left(p\left( x \right)\right)
  = \sum_{j_1=0}^{N_1} \cdots \sum_{j_D=0}^{N_D} c_{\left(j_1, \ldots, j_D\right)}
    \prod_{d=1}^D T_{j_d} \left(p_d\right).
$$
Here, $p_d$ is the $d$-th element of $p\left( x \right)$.


In order to calculate the coefficients $c_{\bar j} = c_{\left(j_1, \ldots, j_D\right)}$ we introduce the multivariate Chebyshev points (of second kind). We consider a multi-index $\bar k$ and set
$$
  q_{\bar k} = \left(q_{k_1},\ldots, q_{k_D} \right) \in \left[-1, 1\right]^D
$$
with
$$
  q_{k_d} = \cos\left(\pi \frac{k_d}{N_d} \right), \;
  0\leq k_d \leq N_d, \;
  d = 1,\ldots,D.
$$
The mapping
$$
  x_{\bar k} = x\left( q_{\bar k} \right) = x\left(q_{k_1},\ldots, q_{k_D}\right)
$$
defines the affine mapping from the standardised domain $\left[-1, 1\right]^D$ to the domain of the  target function.

The coefficients $c_{\bar j} = c_{\left(j_1, \ldots, j_D\right)}$ are given as
$$
  c_{\bar j} = 
  \left( \prod_{d=1}^D \frac{2^{\mathbb{1}_{0<j_d<N_d} }}{n_d} \right)
  \sum_{k_1=0}^{N_1}{´´} \cdots \sum_{k_D=0}^{N_D}{´´} 
  y\left(x\left(q_{k_1},\ldots, q_{k_D}\right)\right)
  \prod_{d=1}^D T_{j_d} \left(q_{k_d}\right).
$$
Here, the notation $\sum{´´}$ represents the weighted sum where the first and last element are assigned weight $\frac{1}{2}$ and all other elements are assigned unit weight.


A first critical aspect of multivariate Chebyshev interpolation is that the method initially requires $\prod_{d=1}^D\left(N_d +1\right)$ evaluations $y\left(x\left( q_{\bar k} \right)\right)$ of the target function at the Chebyshev points $q_{\bar k}$. For larger dimensions (e.g. $D>3$) and computationally expensive target functions this can be a limitation.

Another critical aspect of multivariate Chebyshev interpolation concerns the linear algebra operations. The evaluation of an interpolation $f(x)$ as well as the calibration of each coefficient $c_{\bar j}$ require a calculation of the form
$$
  \sum_{j_1=0}^{N_1} \cdots \sum_{j_D=0}^{N_D} a_{\left(j_1, \ldots, j_D\right)}
  \prod_{d=1}^D b_{j_d}.
$$
A straight forward implementation of such a nested sum involves an iterator along the cartesian product of the indices
$$
  (0,\ldots, N_1), (0,\ldots, N_2), \ldots, (0,\ldots, N_D).
$$
Within each iteration we have $D$ multiplications. This amounts to $D \, \prod_{d=1}^D \left(N_d +1\right)$ multiplications potentially followed by an additions. This illustrates the exponential growth of computational effort in terms of number of dimensions $D$.

In the following sections we will discuss how to implement above nested sum efficiently by exploiting standardised high-level linear algebra operations available in modern programming environments.


# High-level Linear Algebra Operations

In this section we discuss linear algebra operations that turn out to be useful for the implementation of multivariate Chebyshev interpolation. Such operations are often available in linear algebra modules of high-level programming languages. Our example implementations are based on Numpy, TensorFlow and Julia. But we aim at avoiding language or module specific implementation choices.

**Multi-dimensional arrays.**

A guiding principle of our algorithm is the representation of data structures as multi-dimensional arrays. Such multi-dimensional arrays are also called tensors. Tensor operations are further discussed, e.g. in [@GV13], Sec. 12.4.

A $D$-dimensional array ${\cal A} = \left(a_{\bar j}\right)$ is a structure consisting of elements $a_{\bar j} \in \mathbb{R}$ where $\bar j$ is a multi-index $\bar j = \left(j_1, \ldots, j_D\right)$. For each axis (or *mode*) $d=1,\ldots,D$ we have an index range $j_d=1,\ldots,N_d$. The number of dimensions $D$ is also called the order of the tensor.

Obviously, vectors and matrices represent the special cases of one- and two-dimensional arrays or order-1 and order-2 tensors. Scalars can be viewed as order-0 tensors.

The tuple $\left(N_1, \ldots, N_D\right)$ represents the shape of the tensor. The shape specifies the index ranges for each axis.

Elements of a tensor ${\cal A}$ are accessed via the function call operator ${\cal A}\left( \cdot \right)$. That is
$$
  {\cal A}\left(j_1, \ldots, j_D\right) = a_{\left(j_1, \ldots, j_D\right)}.
$$
Sub-tensors or slices are specified by replacing specific indices $j_d$ by "$:$". For example, ${\cal A}\left(:, j_{D-1}, j_D\right)$ is an order-$D-2$ tensor of shape $\left(N_1, \ldots, N_{D-2}\right)$.

**Cartesian product of vectors.**

For a list of vectors $v^1, \ldots, v^D$ with $v^d = \left(v_{j_d}^d \right)_{j_d=1}^{N_d}$ ($d=1,\ldots,D$) we define the cartesian product
$$
  V = \text{product}\left(v^1, \ldots, v^D\right)
$$
as the $\left(\prod_{d=1}^D N_d\right) \times D$-matrix $V$ with elements
$$
  V = \left[ \begin{array}{ccccc}
    v_{1}^{1} & v_{1}^{2} & \ldots & v_{1}^{D-1} & v_{1}^{D} \\
    v_{1}^{1} & v_{1}^{2} & \ldots & v_{1}^{D-1} & v_{2}^{D} \\
              &           & \vdots &             &           \\
    v_{N_1}^{1} & v_{N_2}^{2} & \ldots & v_{N_{D-1}}^{D-1} & v_{N_D-1}^{D} \\
    v_{N_1}^{1} & v_{N_2}^{2} & \ldots & v_{N_{D-1}}^{D-1} & v_{N_D}^{D}
  \end{array} \right].
$$
In this ordering the elements in the last column change fastest and the elements in the first column change slowest.

We will apply the cartesian product operation for real vectors as well as for index vectors. In particular, the cartesian product of indices
$$
  J = \text{product}\left(\left(1,\ldots,N_1\right), \ldots, \left(1,\ldots,N_D\right)\right)
$$
yields a vector of multi-indices $J = \left(\bar j\right)_{\bar j}$ that allows to iterate the elements of a tensor ${\cal A} = \left(a_{\bar j}\right)$.

**Re-shaping tensors.**

Reshaping changes the order of a tensor but keeps the data elements unchanged. The most basic form of re-shaping a tensor is the flattening or vectorisation. We define
$$
 \text{vec}\left({\cal A}\right) =
 \left[ \begin{array}{c}
   a_{\left(1, 1, \ldots, 1, 1\right)} \\
   a_{\left(1, 1, \ldots, 1, 2\right)} \\
   \vdots \\
   a_{\left(N_1, N_2, \ldots, N_{D-1}, N_{D}-1\right)} \\
   a_{\left(N_1, N_2, \ldots, N_{D-1}, N_{D}\right)}
 \end{array} \right]
 = \left[ a_{\bar j} \right]_{\bar j \in J}.
$$
That is, we align the tensor elements with the last axis changing fastest and the first axis changing slowest similarly as in the cartesian product specification.

A general re-shape operation 
$$
  {\cal B} =
  \text{reshape}\left({\cal A}, \left(M_1,\ldots, M_E\right)\right)
$$
of a tensor ${\cal A}$ with shape $\left(N_1,\ldots, N_D\right)$ into a tensor ${\cal B}$ with shape $\left(M_1,\ldots, M_E\right)$ and
$$
  \prod_{d=1}^D N_d = \prod_{e=1}^E M_e
$$
is defined via
$$
  \text{vec}\left({\cal A}\right) = \text{vec}\left({\cal B}\right).
$$

**Element-wise tensor multiplication with broadcasting.**

Element-wise tensor multiplication is used to delegate calculations to efficient low-level implementations utilising e.g. BLAS routines and parallelisation. This approach is particularly efficient when combined with the concept of broadcasting.

Consider two tensors ${\cal A} = \left(a_{\bar j}\right)$ and ${\cal B} = \left(b_{\bar j}\right)$ with shape $\left(N_1, \ldots, N_D\right)$ and $\left(M_1, \ldots, M_D\right)$. We impose the constraint that
$$
  N_d = M_d \; \text{or} \;
  N_d = 1   \; \text{or} \;
  M_d = 1   \; \text{for} \;
  d = 1,\ldots,D.
$$
The element-wise product with broadcasting
$$
  {\cal C} = {\cal A} \; {.*} \; {\cal B}
$$
yields a tensor ${\cal C}$ with shape
$$
  \left(\max\left\{N_1, M_1\right\}, \ldots, \max\left\{N_D, M_D\right\}\right).
$$
The elements $c_{\bar j} = c_{\left(j_1,\ldots,j_D\right)}$ of the resulting tensor ${\cal C}$ are
$$
  c_{\left(j_1,\ldots,j_D\right)} =
  a_{\left(\min\left\{j_1, N_1\right\},\ldots,\min\left\{j_D, N_D\right\}\right)} \cdot
  b_{\left(\min\left\{j_1, M_1\right\},\ldots,\min\left\{j_D, M_D\right\}\right)}.
$$

Element-wise multiplication with broadcasting is the standard behaviour for multiplication of multi-dimensional arrays in Numpy and TensorFlow. In Julia it is implemented by the ".*" operator.

**Generalised matrix multiplication.**

The Python Enhancement Proposal (PEP) 465 [@Smi14] specifies a matrix multiplication that generalises to multi-dimensional arrays. This operation is implemented in Numpy and TensorFlow as the *matmul* function.

Suppose, we have two tensors ${\cal A}$ and ${\cal B}$ with shape $\left(N_1, \ldots, N_D\right)$ and $\left(M_1, \ldots, M_D\right)$. We require that $D\geq 2$,
$$
  N_d = M_d \; \text{or} \;
  N_d = 1   \; \text{or} \;
  M_d = 1   \; \text{for} \;
  d = 1,\ldots,D-2,
$$
and
$$
  N_D = M_{D-1}.
$$
The generalised matrix multiplication is defined as
$$
  {\cal C} = \text{matmul}\left({\cal A}, {\cal B}\right). 
$$
The result tensor ${\cal C}$ is of shape
$$
  \left(\max\left\{N_1, M_1\right\}, \ldots, \max\left\{N_{D-2}, M_{D-2}\right\},
  N_{D-1}, M_D
  \right).
$$
And the elements of ${\cal C}$ are calculated as
$$
  {\cal C}\left(:, i, j\right) =
  \sum_{k=1}^{N_D} {\cal A}\left(:, i, k\right) \; {.*} \; {\cal B}\left(:, k, j\right)
$$
for $i=1,\ldots,N_{d-1}$ and $j=1,\ldots,M_d$. Here, ${\cal C}\left(:, i, j\right)$ is the tensor of order $D-2$ where we fix the last two axes of ${\cal C}$. Analogously, ${\cal A}\left(:, i, k\right)$ and ${\cal B}\left(:, k, j\right)$ are specified.

We note that the generalised matrix multiplication can be related to the *modal product* of tensors and matrices. Consider a matrix ${\cal M}$ of shape $\left(M_1, M_2\right)$ with $M_2 = N_d$. The mode-$d$ product
$$
  {\cal C} = {\cal A} \times_d {\cal M}
$$
yields a tensor of shape
$$
  \left(N_1, \ldots, N_{d-1}, M_1, N_{d+1}, N_{D}\right).
$$
The elements of ${\cal C} = \left(c_{\left(j_1, \ldots,j_D\right)} \right)$ are calculated as
$$
  {\cal C}{\left(j_1, \ldots,j_{d-1}, i, j_{d+1}, j_D \right)} =
  \sum_{k=1}^{N_d} {\cal M}\left(i, k\right) {\cal A}{\left(j_1, \ldots,j_{d-1}, k, j_{d+1}, j_D \right)}
$$
for $i=1,\ldots,M_1$.

It turns out that the mode-$D$ product along the last axis is
$$
  {\cal A} \times_D {\cal M} =
  \text{matmul}\left({\cal A},
  \text{reshape}\left({\cal M}^\top, \left(1,\ldots,1,M_2,M_1\right)\right)\right).
$$


# Reformulated Chebyshev Interpolation

We return to the task of calculating nested sums of the form
$$
  \sum_{j_1=0}^{N_1} \cdots \sum_{j_D=0}^{N_D} a_{\left(j_1, \ldots, j_D\right)}
  \prod_{d=1}^D b_{j_d}.
$$
The coefficients $a_{\left(j_1, \ldots, j_D\right)}$ can be aligned in an order-$D$ tensor ${\cal A}$.  of shape $\left(N_1+1, \ldots, N_D + 1\right)$. Similarly, the Chebyshev polynomial values $b_{j_d}$ can
be arranged as $D$ matrices ${\cal B}^d$ of shape $\left(1, N_d+1\right)$.

With this notation the nested sum becomes a sequence of modal products
$$
  \sum_{j_1=0}^{N_1} \cdots \sum_{j_D=0}^{N_D} a_{\left(j_1, \ldots, j_D\right)}
  \prod_{d=1}^D b_{j_d}
  = {\cal C}\left(1, \ldots, 1\right)
$$
where the order $D$ tensor ${\cal C}$ with shape $\left(1, \ldots, 1\right)$ is
$$
  {\cal C} = \left(\left({\cal A} \times_D {\cal B}^D\right) \times_{D-1} \ldots \right) \times_1 {\cal B}^1.
$$

The property that the multivariate Chebyshev interpolation formula can be written as modal product is also observed in [@Poe20], sec. 5.1. We also note that the sequence of modal products is invariant with respect to its ordering. See [@GV13], Theorem 12.4.1. Thus, we could also calculate
$$
  {\cal C} = \left(\left({\cal A} \times_1 {\cal B}^1\right) \times_{2} \ldots \right) \times_D {\cal B}^D.  
$$

**Chebyshev batch calculation**

The interpolation function $f$ often needs to be evaluated for various inputs $x^1, \ldots, x^N$ (without subscript). In such a context we call $N$ the batch size for evaluation. In order to utilize BLAS routines and parallelisation we want to avoid manual iteration over the elements of a batch. Instead, we carefully use broadcasting to vectorise calculations.

Input to the Chebyshev batch calculation are a matrix ${\cal P}$ and an order-$D$ tensor ${\cal C}$. The matrix ${\cal P}$ is of shape $\left(D, N\right)$ and consists of points from the standardised domain $\left[-1, 1\right]^D$. That is,
$$
  {\cal P} = \left[ p\left(x^1\right), \ldots, p\left(x^N\right) \right].
$$
The tensor ${\cal C}$ is of shape $\left(N_1+1, \ldots, N_D+1\right)$ and for the usage of interpolation consists of the Chebyshev coefficients $c_{\bar j}$.

For each row $d$ and inputs ${\cal P}\left(d,:\right)$ we can calculate a matrix of Chebyshev polynomial values
${\cal T}^d$ of shape $\left(N_d+1, N\right)$ with entries
$$
\begin{aligned}
  {\cal T}^d\left(1,:\right) &= \left(1, \ldots,1\right), \\
  {\cal T}^d\left(2,:\right) &= {\cal P}\left(d,:\right), \\
  {\cal T}^d\left(j,:\right) &= 2 \, {\cal P}\left(d,:\right) \; {.*} \; {\cal T}^d\left(j-1,:\right) - {\cal T}^d\left(j-2,:\right),
\end{aligned}
$$
for $j=3,\ldots,N_d+1$.

With the following algorithm we calculate a vector
$$
  {\cal R} = \left[ f\left(x^1\right), \ldots, f\left(x^N\right) \right]^\top.
$$

Start:

1. Initialise ${\cal R} \leftarrow \text{reshape}\left({\cal C}, \left(1, N_1+1,\ldots, N_D+1\right) \right)$ by adding a trivial first axis.

2. For $d=D, D-1, \ldots, 1$:

   2.1. Calculate Chebyshev matrix ${\cal T}^d$ of shape $\left(N_d+1, N\right)$.

   2.2. Re-arrange ${\cal T}^d \leftarrow \text{reshape}\left({{\cal T}^d}^\top, \left(N,\underbrace{1, \ldots, 1}_{k \text{ times}}, N_d+1, 1\right) \right)$ where $k=\max\left\{d-2, 0\right\}$.

   2.3. If $d=1$, i.e. ${\cal R}$ is an order-2 tensor of shape $\left(R_1,R_2\right)$ (last iteration):

     2.3.1. Adjust ${\cal R} \leftarrow \text{reshape}\left({\cal R}, \left(R_1, 1, R_2\right) \right)$.

   2.4. Calculate ${\cal R} \leftarrow \text{matmul}\left({\cal R}, {\cal T}^d\right)$. If $d>1$ this step yields an order-$(d+1)$ tensor of shape $\left(N, N_1, \ldots, N_{d-1}, 1 \right)$. In the last iteration with $d=1$ we get an order-$3$ tensor of shape $\left(N, 1, 1 \right)$.

   2.5. Remove trivial last axis via
   
     2.5.1. ${\cal R} \leftarrow \text{reshape}\left({\cal R}, \left(N, N_1,\ldots, N_{d-1} \right)\right)$ if $d>1$ or

     2.5.1. ${\cal R} \leftarrow \text{reshape}\left({\cal R}, \left(N, 1 \right)\right)$ if $d=1$.
   
3. Remove the remaining trivial axis ${\cal R} \leftarrow \text{reshape}\left({\cal R}, \left(1 \right)\right)$.

4. Return the vector ${\cal R}$ as result.
 
Finish.

We assess the computational effort of the Chebyshev batch calculation. Re-shape and matrix transposition operations are cheap because they do not require data access or data manipulation. We count multiplications which may be followed by an addition as a single operation.

Chebyshev matrix calculation in step 2.1. amounts to
$$
  N \sum_{d=1}^D \left(N_d - 1 \right)
$$
operations. This is more or less ${\cal O}\left(N\right)$ and relatively cheap.

The computationally expensive step is the generalised matrix multiplication in step 2.4. Here we count
$$
  N \sum_{d-1}^D \left(\prod_{j=1}^{d-1} \left(N_j + 1\right) \right) N_d
  =
  N \sum_{d-1}^D \, \prod_{j=1}^{d} \left(N_j + 1\right)
$$
operations. For equal degrees $\bar N = N_1 = \ldots = N_D \geq 1$ we get the estimate
$$
  N \sum_{d-1}^D \left(\bar N + 1\right)^d
  < 2 N \left({\bar N} + 1\right)^D.
$$
The proposed algorithm still suffers from the exponential growth in the number of dimensions $D$. However, we save a factor of $D/2$ compared to a standard implementation via cartesian product.




# Case Study: Implied Volatility Surface of Heston Model

- Heston model and model parameters

- Vanilla option pricing via QuantLib

- smile parametrisation

- Chebyshev parametrisation: parameter ranges, degrees, resulting number of coefficients

- smile and term structure approximation for various parameter scenarios

- (quasi) random sample test(s)

- limitation: extrapolation

# References

<div id="refs"></div>

