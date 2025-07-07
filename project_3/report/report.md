---
title: Solving the One-Dimensional Diffusion Equation using Feedforward Neural Networks
subtitle: FYS-STK4155 - Project 3
authors:
  - name: Insert Name
site:
  template: article-theme
exports:
  - format: pdf
    template: ../report_template
    output: report.pdf
    showtoc: true
math:
  # Note the 'single quotes'
  '\argmin': '\operatorname{argmin}'
  '\drm': '\mathrm{d}'
  '\N': '\mathbb{N}'
  '\R': '\mathbb{R}'
  '\Set': '{\left\{ #1 \right\}}'
bibliography: references.bib
abstract: |
  This project studies the use of feedforward neural networks (FFNN) to solve the one-dimensional heat equation with Dirichlet boundary conditions. Through hyperparameter tuning using grid search, it was found that FFNN models with three hidden layers converged effectively to the exact solution. Among various activation functions, the Sigmoid Linear Unit (SiLU) provided the best performance, followed by the tanh function, which offered a lower computational cost. The stability of the FFNN model was compared to the explicit forward Euler scheme, and it was found to perform similarly to an Euler model with $\Delta x = 0.1$, although with a more irregular error distribution. To improve the convergence of the FFNN model, training on additional points or the application of more advanced optimization techniques may be necessary. This work demonstrates the potential of neural networks for solving differential equations, especially in dynamic systems, while acknowledging challenges related to training stability and computational efficiency.
---

# Introduction

The goal of this project is to explore the use of feedforward neural networks (FFNN) for solving the one-dimensional heat equation with Dirichlet boundary conditions. Specifically, we investigate how neural networks can approximate the solution to this equation and compare their performance to classical methods such as the explicit forward Euler scheme. Through grid search, we aim to identify optimal network architectures and activation functions for this task, evaluate the stability and accuracy of the FFNN approach, and assess its potential for solving PDEs in practical applications. By comparing the results from neural network-based solutions and traditional numerical methods, this study seeks to provide insights into the advantages and limitations of using machine learning for solving differential equations.

# Theory and Method

## One-Dimensional Heat Equation

The diffusion equation is a parabolic partial differential equation (PDE) of the form 

$$
  \frac{\partial u(\mathbf{r}, t)}{\partial t} = \alpha \nabla^2 u(\mathbf{r}, t)
$$

for constant diffusion coefficient $\alpha\in\R$. It describes the macroscopic distribution of a given quantity $u$ over time as a result of a microscopic diffusion process, such as Brownian motion or heat transfer. In this particular case, we are interested in modeling the time-evolving heat distribution of a one-dimensional rod of unit length $\ell = 1$. At time $t > 0$, the temperature of the rod at position $x\in[0,\ell]$ can be described by a function $u:[0,\ell]\times [0,\infty) \to\R$ that satisfies the one-dimensional diffusion equation (also known as the heat equation)

$$
\label{equation:1d-diffusion}
\begin{split}
  \frac{\partial^2 u(x,t)}{\partial x^2} =& \alpha \frac{\partial u(x,t)}{\partial t},\; t > 0, x\in [0,\ell], \\
  u_{xx} =& \alpha u_t
\end{split}
$$

where $\alpha\in\R$ is the thermal diffusivity of the rod. For simplicity, we assume $\alpha = 1$ from hereon. Furthermore, we assume that $u$ satisfies the initial conditions

$$
\label{equation:1d-diffusion-initial-conditions}
  u(x, 0) = \phi(x),\; x\in(0, \ell)
$$

where $\phi(x):[0,\ell]\to\R$ represents the inital heat distribution. We also assume that $u$ satisfies the Dirichlet boundary conditions

$$
\label{equation:1d-diffusion-boundary-conditions}
  u(0, t) = u(\ell, t) = 0, \quad t \geq 0.
$$

### Analytical Solution

To solve the heat equation analytically, we observe that both the diffusion equation [](#equation:1d-diffusion) and the boundary conditions [](#equation:1d-diffusion-boundary-conditions) are linear and homogenous. This allows us to solve the equation using separation of variables. Assuming a solution of the form $u(x, t) = X(x)T(t)$, we get the following differential equation

$$
\begin{align*}
  \frac{\partial}{\partial t} (XT) =& \frac{\partial^2}{\partial x^2} (XT) \\
  X \frac{\drm T}{\drm t} =& T \frac{\drm^2 X}{\drm x^2} \\
  X\dot{T} =& T X''.
\end{align*}
$$

Dividing both sides with $XT$, we obtain

$$
  \frac{1}{T} \dot{T} = \frac{1}{X} X'' = -\lambda,
$$

which is satisfied if and only if both sides are equal to a constant $\lambda\in\R$. This can be separated into a temporal equation

$$
\label{equation:temporal}
  \dot{T} = -\lambda T
$$

and a spatial equation

$$
\label{equation:spatial}
  \quad X'' = -\lambda X
$$

##### Satisfying the Boundary Conditions
We require that [](#equation:spatial), which is a second order linear ordinary differential equation (ODE), satisfies the boundary condition

$$
\label{equation:spatial-boundary-conditions}
  X(0) = X(\ell) = 0.
$$

The spatial equation [](#equation:spatial) together with the boundary conditions define an eigenvalue problem, where $\lambda$ are eigenvalues of the second order differential operator $\drm^2/\drm x^2$, and $X$ are the corresponding eigenfunctions. The only non-trivial solution to [](#equation:spatial) occurs for eigenvalues $\lambda > 0$, in which case $X$ has solutions of the form

$$
  X(x) = B \sin(\sqrt{\lambda}x) + C\cos(\sqrt{\lambda}x) 
$$

From the boundary conditions [](#equation:spatial-boundary-conditions), we get

$$
\begin{gather}
  X(0) = 0 \implies C = 0 \\
  X(\ell) = 0 \implies \sin(\sqrt{\lambda} \ell) = 0 \implies \sqrt{\lambda} = \frac{n\pi}{\ell}, n\in\N_+
\end{gather}
$$

Thus, [](#equation:spatial) has solutions of the form

$$
\label{equation:spatial-solution}
  X_n (x) = B_n \sin\left(\frac{n\pi}{\ell} \right),\; n\in\N_+
$$

The temporal equation [](#equation:temporal) is a first order linear ODE with general solutions

$$
  T_n(t) = A_n e^{-\lambda_n t} = A_n e^{-(n\pi/\ell)^2 t},\; n\in\N_+
$$

Thus, for every eigenvalue $\lambda_n = (n\pi/\ell)^2$ with corresponding eigenfunction $X_n$, there is a solution $T_n$ such that the function 

$$
  u_n (x,t) = T_n (t) X_n = D_n \sin\left(\frac{n\pi}{\ell}\right) e^{-(n\pi/\ell)^2 t},\; D_n = A_n B_n
$$

solves the diffusion equation [](#equation:1d-diffusion) with Dirichlet boundary conditions. Due to the linearity of the diffusion equation, every linear combination of $u_n$ for $n\in\N_+$ is also a solution. The general solution can therefore be written as an infinite series of the form

$$
  u(x, t) = \sum_{n=1}^\infty u_n (x,t)
$$

#### Satisfying the Inital Conditions

From the initial conditions [](#equation:1d-diffusion-initial-conditions), we require that

$$
\label{equation:1d-diffusion-coefficient}
  u(x, 0) = \sum_{n=1}^\infty D_n X_n (x) = \phi(x),\; x\in(0, \ell)
$$

To find the cofficients $D_n$, we use the fact that the eigenfunctions $X_n$ form an orthogonal basis for the Hilbert space $L^2 ([0,\ell])$ with the inner product

$$
  \langle f, g \rangle = \int_0^\ell f(x) g(x)\;\drm x, \; f,g\in L^2 ([0,\ell])
$$

Multiplying [](#equation:1d-diffusion-coefficient) by $X_n$ for a fixed $n\in\N_+$ and integrating over $[0,\ell]$, we find that the coefficients are given by the formula

$$
  D_n \langle X_n, X_n \rangle = \langle X_n, \phi \rangle \implies D_n = \frac{\langle X_n, \phi\rangle}{\langle X_n, X_n \rangle}.
$$

Substituting [](#equation:spatial-solution) into the formula gives

$$
  D_n = \frac{\langle X_n, \phi\rangle}{\langle X_n, X_n \rangle} = \frac{\int_0^\ell \sin(n\pi x/ \ell) \phi(x)\;\drm x}{\int_0^\ell \sin^2 (n\pi x / \ell)\;\drm x} = \frac{2}{\ell} \int_0^\ell \sin\left(\frac{n\pi}{\ell}\right) \phi(x)\;\drm x,
$$

which are recognized as the Fourier coefficients.

#### Sinusoidal Initial Condition

In this study, we have have used sinusoidal initial condition $u(x,0) = \sin\left(\frac{\pi}{\ell} x\right)$, which gives the following coefficients for the general solution:

$$
\begin{align*}
  D_n = \frac{2}{\ell} \int_0^\ell \sin\left(\frac{n\pi}{\ell} x\right) \sin\left(\frac{\pi}{\ell} x\right)\;\drm x = \left\langle \sin\left(\frac{n\pi}{\ell} x\right), \sin\left(\frac{\pi}{\ell} x\right) \right\rangle = \begin{cases} 1,\quad& n = 1 \\ 0, \quad& n \geq 2 \end{cases} 
\end{align*}
$$

Thus, the analytical solution reduces to $u(x,t) = \sin\left(\frac{\pi}{\ell} x \right) e^{-(n\pi/\ell)^2 t}$. A plot of the solution is given in [](#figure:1d-diffusion-solution-sinus). 

```{figure} figures/1d_diffusion_solution_sinus.pdf
:label: figure:1d-diffusion-solution-sinus
:alt: Activation Functions.
:align: center

Plot of the analytical solution to the one dimensional heat equation for $x\in[0,1]$ and $t\in[0,1]$ with Dirichlet boundary conditions and intial conditions $u(x,0) = \sin(\pi x)$.
```

### Explicit Forward Euler Scheme

This derivation of the explicit forward Euler scheme for the one-dimensional diffusion equation is based on @note_hjortjensen_2015. We start by discretizing the spatial and temporal variables. The space domain is divided into $n\in\N_+$ points with step given $\Delta x = \ell/(n - 1)$. The time step $\Delta t$, on the other hand, is set independently and is constrained by a stability criterion given below. This discretization forms a two-dimensional grid where the spatial points $x_i$ and time points $t_j$ are given by

$$
\begin{align*}
  x_i =& i\Delta x,\; 0 \leq i \leq n + 1 \\
  t_j =& j\Delta t,\; j \geq 0.
\end{align*}
$$

Furthermore, we approximate $u_t$ by the forward difference

$$
  u_t = \frac{u(x,t + \Delta t) - u(x,t)}{\Delta t} + \mathcal{O}(\Delta t) = \frac{u(x_i, t_j + \Delta t) - u(x_i, t_j)}{\Delta t} + \mathcal{O}(\Delta t),
$$

where $\mathcal{O}(\Delta t)$ is the truncation error. Likewise, we approximate $u_{xx}$ by the second order central difference

$$
\begin{align*}
  u_{xx} =& \frac{u(x + \Delta x, t) - 2u(x, t) + u(x - \Delta x, t)}{\Delta x^2} + \mathcal{O}(\Delta x^2) \\
  =& \frac{u(x_i + \Delta x, t_j) - 2u(x_i, t_j) + u(x_i - \Delta x, t_j)}{\Delta x^2} + \mathcal{O}(\Delta x^2)
\end{align*}
$$

Inserting $u_t$ and $u_{xx}$ into [](#equation:1d-diffusion) gives

$$
\begin{align*}
  \frac{u(x_i, t_j + \Delta t) - u(x_i, t_j)}{\Delta t} =& \frac{u(x_i + \Delta x, t_j) - 2u(x_i, t_j) + u(x_i - \Delta x, t_j)}{\Delta x^2} + \mathcal{O}(\Delta x^2) \\
  \frac{u_{i,j + 1} - u_{i,j}}{\Delta t} =& \frac{u_{i+1, j} - 2u_{i,j} + u_{i-1, j}}{\Delta x^2}. 
\end{align*}
$$

Defining $\rho = \Delta t / \Delta x^2$ yields the explicit scheme

$$
  u_{i,j+1} = \rho u_{i-1, j} + (1 - 2\rho)u_{i,j} + \rho u_{i+1, j}
$$

with stability condition

$$
\label{equation:1d-diffusion-euler-stability-criterion}
  \rho \leq 1/2
$$

### Feedforward Neural Network Solver

In order to solve the one-dimensional diffusion equation using a feedforward neural network (FFNN), we construct a possible trial solution for [](#equation:1d-diffusion) of the form [@note_hjortjensen_2023]

$$
  \hat{u} (x, t; P) = h_1 (x, t) + h_2 [x, t, N(x, t; P)],
$$

where $N$ is the output from an FFNN model with set of biases and weights $P$. In order to satisfy the boundary [](#equation:1d-diffusion-boundary-conditions) and initial [](#equation:1d-diffusion-initial-conditions) conditions, we require that 

$$
\begin{equation}
\begin{aligned}
  h_1 (0, t) = h_1 (\ell, t) = 0 \\
  h_1 (x, 0) = \sin\left(\frac{\pi}{\ell} x\right)
\end{aligned}
\quad
\begin{aligned}
  h_2 (0,t) = h_2 (\ell, t) = 0 \\
  h_2 (x, 0) = 0.
\end{aligned}
\end{equation}
$$

Possible solutions satisfying these conditions are $h_2 (x, t) = \frac{x}{\ell}(1 - \frac{x}{\ell})t N(x, t; P)$ and

$$
\begin{align*}
  h_1 (x, t) =& (1 - t)\left(u(x, 0) - \left[\left(1 - \frac{x}{\ell}\right)u(0, 0) + \frac{x}{\ell} u(0, 0)\right]\right) \\
  =& (1 - t) u(x, 0) = (1 - t)\sin\left(\frac{\pi}{\ell} x\right),
\end{align*}
$$

giving

$$
  \hat{u} (x,t; P) = (1 - t)\sin\left(\frac{\pi}{\ell}x\right) + \frac{x}{\ell}\left(1 - \frac{x}{\ell}\right)t N(x, t; P).
$$

The optimal trial solution is given by model parameters $P$ that minimize the resdiuals of the heat equation [](#equation:1d-diffusion). At a single point $(x, t)\in [0,\ell]\times[0,\infty)$, we must therefore solve the following minimization problem:

$$
  \argmin_{P} \left(\frac{\partial}{\partial t} \hat{u}(x, t; P) - \frac{\partial^2}{\partial x^2} \hat{u}(x, t; P) \right)^2.
$$

If we evaluate the FFNN model over a set of $N\in\N_+$ points, $\Set{(x_n, t_n)}_{n=1}^N$, the total cost we need to minimize is given by

$$
  C(x,t; P) = \frac{1}{N_x N_t} \sum_{i=1}^{N_x} \sum_{j=1}^{N_t} \left(\frac{\partial}{\partial t} \hat{u}(x_i, t_j; P) - \frac{\partial^2}{\partial x^2} \hat{u}(x_i, t_j; P) \right).
$$

# Results

## Explicit Euler Scheme

We investigated finite difference solutions to the one-dimensional heat equation using the explicit forward Euler method. The experiments were conducted with spatial intervals $\Delta x = 0.1$ and $\Delta x = 0.01$, where the time step $\Delta t$ was determined according to the stability criterion [](#equation:1d-diffusion-euler-stability-criterion).

```{figure} figures/1d_diffusion_euler_absolute_error.pdf
:label: figure:1d-diffusion-euler-absolute-error
:alt: Explicit Euler Scheme Absolute Error.
:align: center

Heatmaps showing the absolute error of the forward Euler approximations to the one-dimensional heat equation: (left) for a spatial interval of $\Delta x = 0.1$, and (right) for $\Delta x = 0.01$.
```

[](#figure:1d-diffusion-euler-absolute-error) illustrates the absolute error of the Euler approximations compared to the analytical solution. Generally, the error is initially large but decreases over time, indicating that the approximation stabilizes as the system evolves toward a stationary state. The results also show that decreasing $\Delta x$ significantly reduces the error. However, this improvement is limited by the stability criterion as $\Delta t$ must be reduced accordingly, which can lead to numerical instability if the time step is too small for the chosen $\Delta x$.

[](#figure:1d-diffusion-euler-snapshots) shows the explicit Euler solutions for $\Delta x = 0.1$ at times $t = 0.1$ and $t = 0.6$. In these plots, we observe that the numerical solution tends to underestimate the exact solution, exhibiting an undershooting error. Over time, as the heat distribution becomes more uniform, the error diminishes. 

```{figure} figures/1d_diffusion_euler_snapshots.pdf
:label: figure:1d-diffusion-euler-snapshots
:alt: Explicit Euler Scheme.
:align: center

Plots of explicit Euler approximations with step $\Delta x = 0.1$ to the one-dimensional heat equation at times $t=0.1$ (left) and $t=0.6$ (right), compared with the exact solutions.
```

## Feedforward Neural Network Solver

### Complexity Grid Search

We performed a grid search to examine stability of the neural PDE solver as functions of hidden layers and number of nodes. The search was conducted over combinations of $1$ to $3$ hidden layers, and neuron configurations ranging from $2 = 8$ to $2 = 128$. The search utilized the sigmoid activation function, with a training setup consisting of $10$ spatial and $10$ temporal points. Training was performed for $1000$ epochs. The results of the grid search are presented in [](#figure:1d-diffusion-neurons-layers-loss) and [](#table:1d-diffusion-grid-search).

The grid search results indicate that the stability of the neural PDE solver generally improves as the model complexity increases. Specifically, three-layer models tend to outperform those with fewer layers, except in cases where the models have a very low number of neurons per layer. This trend is expected, as the dynamic nature of the heat equation requires a sufficient level of model complexity to capture its behavior. However, this increased complexity comes with the trade-off of longer training times. Finally, no clear trend emerges regarding the optimal number of neurons per layer or the arrangement of high versus low neuron layers.

```{figure} figures/1d_diffusion_neurons_layers_loss.pdf
:label: figure:1d-diffusion-neurons-layers-loss
:alt: Grid search result.
:align: center

Scatter plot showing the final loss of the neural PDE solver as a function of the total number of neurons and the number of hidden layers. The color of each point represents the corresponding PDE residual loss, with lower values indicating better performance.
```

:::{table} Top ten performing hidden layer configurations from the grid search of the neural PDE solver, ranked by final loss.
:label: table:1d-diffusion-grid-search
:align: center

| Hidden layers | Final loss | Training time [s] |
|:--|:--|--:|
| (32, 128, 64)  | $\num{1.73e-01}$ | $2.48$ |
| (128, 32, 64)  | $\num{1.81e-01}$ | $1.77$ |
| (128, 128, 64) | $\num{1.89e-01}$ | $3.77$ |
| (16, 128, 64)  | $\num{2.00e-01}$ | $2.31$ |
| (64, 32, 64)   | $\num{2.31e-01}$ | $1.50$ |
| (32, 32, 64)   | $\num{2.32e-01}$ | $1.34$ |
| (32, 64, 64)   | $\num{2.32e-01}$ | $1.55$ |
| (64, 128, 64)  | $\num{2.39e-01}$ | $2.87$ |
| (128, 16, 64)  | $\num{2.44e-01}$ | $1.63$ |
| (8, 128, 64)   | $\num{2.51e-01}$ | $1.82$ |
:::

### Comparison of Activation Function

Following the grid search, we evaluated the stability of the neural PDE solver with different activation functions. The tests were conducted using the hidden layer configuration $(32, 128, 64)$, which was selected as a balanced choice, offering a good trade-off between performance and training time. The results are summarized in [](#table:1d-diffusion-activation). 

Interestingly, the sigmoid activation function performed relatively well, achieving a moderate final loss. This suggests that the sigmoid function is not prone to the vanishing gradient problem in this particular setup, which is often a concern in deeper networks. In contrast, the rectified linear unit (ReLU) function performed poorly, yielding a significantly higher final loss, indicating that it is unsuitable for approximating the heat equation.

Leaky ReLU showed an improvement over ReLU as expected, but still lagged behind other activation functions in terms of stability and final loss. While it helps to alleviate the issue of dying neurons, it was not as effective as more complex activations.

The exponential linear unit (ELU) performed comparatively to the sigmoid function, suggesting it is more suitable than (Leaky) ReLU to approximate the heat equation. However, its performance was still worse than the hyperbolic tangent (tanh) function, which achieved lower loss at comparable computation cost. Lastly, the Sigmoid Linear Unit (SiLU) activation function, also known as Swish, yielded the best performance overall, achieving the lowest final loss. However, it comes at the cost of slightly longer training times compared with tanh.

:::{table} Comparison of activation functions used in the neural PDE solver, showing final loss values and corresponding training times. The models were trained with the hidden layer configuration $(32, 128, 64)$ for each activation function.
:label: table:1d-diffusion-activation
:align: center

| Activation | Final Loss | Training time |
|:--|:--|:--|
| SiLU | $\num{1.28e-03}$ | $3.34$ |
| tanh | $\num{1.84e-03}$ | $2.91$ |
| sigmoid | $\num{2.44e-03}$ | $2.76$ |
| ELU ($\alpha=1$) | $\num{2.85e-03}$ | $2.97$ |
| Leaky ReLU | $\num{1.44e-01}$ | $1.21$ |
| ReLU | $\num{1.21e+00}$ | $1.12$ |
:::

## Comparison of Euler Scheme and Neural Network Solver

We compared the stability and performance of the Euler scheme and the neural network solver for the one-dimensional heat equation. The Euler solution was computed with a spatial step size of $\Delta x = 0.01$ and a time step of $\Delta t = \num{5e-5}$. The neural network solver, on the other hand, was trained using a hidden layer configuration of $(32, 128, 64)$ with the SiLU activation function. The comparison of the results is presented in [](#figure:1d-diffusion_euler_neural_absolute_error). The Euler method generally outperforms the neural network solver in terms of accuracy. Interestingly, the error level of the neural solver is comparable to that of the Euler method when using a larger spatial step size of $\Delta x = 0.1$. Additionally, the neural network solver exhibits a slightly more fluctuating error distribution, indicating potential sensitivity to the model's complexity or training process.

```{figure} figures/1d_diffusion_euler_neural_absolute_error.pdf
:label: figure:1d-diffusion_euler_neural_absolute_error
:alt: Explicit Euler Scheme and Neural Solver Absolute Error.
:align: center

Heatmaps comparing the absolute errors of solutions to the one-dimensional heat equation obtained using the explicit Euler scheme (left) and a neural network solver (right).
```

# Discussion

In this section we give a critical discussion of the two approaches for solving differential equation examined in this report: explicit forward Euler scheme and feedforward neural networks.

One clear advantage of neural networks is their ability to approximate solutions over continuous domains. This makes neural models more flexible, as they can potentially capture complex, non-linear dynamics. In contrast, Euler's method is inherently discrete, as it approximates the solution by stepping through time and space at fixed intervals. This discretization can lead to poor approximations for regions with sharp gradients or highly localized phenomena. Neural models, by their continuous nature, allow for capturing such intricate behaviors, provided sufficient training data and model complexity.

A key strength of the Euler method lies in its well-understood stability conditions and computational efficiency. This predictability makes the method more reliable and easier to implement, particularly for simpler problems. However, as a first-order method, Euler's method can struggle to maintain high accuracy for nonlinear equations, leading to significant approximation errors. To achieve higher accuracy, smaller time steps or more sophisticated methods, such as implicit schemes, may be required. However, these approaches typically come with increased computational cost.

Neural networks, on the other hand, present challenges in training to effectively capture complex dynamics. Understanding how a network arrives at a particular solution is more opaque compared to finite difference methods, where the process is more straightforward and interpretable. The stability conditions for neural networks are not as well-defined, and ensuring convergence to the true solution requires extensive optimization, including the selection of an optimal network structure, activation functions, and hyperparameters. Additionally, the training process for neural networks can be computationally demanding and time-consuming, particularly for more complex systems. This often involves navigating a vast search space for model configurations, making it difficult to guarantee optimal performance without significant computational resources and experimentation.

# Conclusion

This study has explored the use of feedforward neural networks (FFNN) to solve the one-dimensional heat equation with Dirichlet boundary conditions. Through hyperparameter tuning via grid search, it was found that models with three hidden layers trained on $10$ spatial points and $10$ time points converged sufficiently to the exact solution. Among the activation functions tested, the Sigmoid Linear Unit (SiLU) outperformed others, with the tanh function showing slightly lower performance but offering reduced computational cost. In terms of stability, the FFNN model performed comparably to the explicit forward Euler scheme with a step size of $\Delta x = 0.1$. However, the FFNN exhibited a more irregular error distribution compared to the smoother error profile of the Euler method. To enhance the convergence and accuracy of the FFNN model, further improvements may be necessary, including training on additional points or employing advanced optimization techniques. This study highlights the potential of neural networks for solving differential equations, particularly for more complex, dynamic systems, though challenges such as training stability and computational cost remain.

# Appendix

## Code Repository

The Julia source code used to generate the test results is available at [https://github.com/semapheur/fys-stk4155](https://github.com/semapheur/fys-stk4155).

## Dirichlet Eigenvalue Problem

Consider the eigenvalue problem $X'' = -\lambda X$ for $x\in[0,\ell]$. Here we show that only the trivial solution $X=0$ satisfies the Dirichlet boundary condition $X(0) = 0 = X(\ell)$ for $\lambda \leq 0$.

If $\lambda = 0$, the general solution is $X(x) = A + Bx$. From the boundary conditions, we have

$$
\begin{align*}
  X(0) = 0 \implies& A = 0 \\
  X(\ell) = 0 \implies& B = 0.
\end{align*}
$$

If $\lambda = -\gamma^2 < 0$, the general solution takes the form

$$
  X(x) = C\cosh(\gamma x) + D\sinh(\gamma x)
$$

The boundary conditions require that

$$
\begin{align*}
  X(0) = 0 \implies& C = 0 \\
  X(\ell) = 0 \implies D = 0.
\end{align*}
$$

## Orthogonality Property for Symmetric Eigenfunctions

Recall that that two square integrable functions $f,g\in L([0,\ell])$ are *orthogonal* if

$$
  \langle f, g \rangle = \int_0^\ell f(x) g(x)\;\drm x = 0.
$$

Boundary conditions are *symmetric* if for $f, g\in L^2 ([0,\ell])$

$$
  [f'(x)g(x) - f(x)g'(x)]_{x=a}^{x=b} = 0
$$

Next, we show that if $X_n, X_m$ are two distinct eigenfunctions to the eigenvalue problem $X'' = -\lambda X$ with the Dirichlet boundary conditions, then $X_n$ and $X_m$ are orthogonal. Multiplying $X''_n = -\lambda_n X_n$ with $X_m$ and integrating by parts over $[0,\ell]$ gives

$$
\begin{align*}
  \lambda_n \int_a^b X_n (x) X_m (x) \; \drm x =& -\int_0^\ell X''_n (x) X_m (x) \; \drm x \\
  =& \int_0^\ell X'_n (x) X'_m (x) \;\drm x - [X'_n (x) - X_m (x)]_{x=0}^{x=\ell} \\
  =& -\int_0^\ell X_n (x) \underbrace{X''_m (x)}_{=-\lambda_m X_m} \;\drm x + \underbrace{[X_n (x) X'_m (x) - X'_n (x) X_m (x)]_0^\ell}_{=0} \\
  =& -\lambda_m \int_0^\ell X_n X_m \;\drm x,
\end{align*}
$$

where we have used the fact that $X_n$ and $X_m$ are symmetric with the Dirichlet boundary conditions. Thus,

$$
  (\lambda_n - \lambda_m) \int_0^\ell X_n X_m \;\drm x = 0,
$$

but $\lambda_n \neq\lambda_m$ by assumption. Hence, we conclude that

$$
   \int_0^\ell X_n X_m \;\drm x = \langle X_n, X_m \rangle = 0
$$

as claimed.

## Coefficients for Sinusoidal Initial Conditions

To find the coefficients $D_n$ using sinusoidal initial condition $u(x,0) = \sin\left(\frac{\pi}{\ell} x\right)$, we use the trigonometric identity $\sin(a)\sin(b) = \frac{1}{2}(\cos(a - b) - \cos(a + b))$ to get

$$
\begin{align*}
  D_n =& \frac{2}{\ell} \int_0^\ell \sin\left(\frac{n\pi}{\ell} x \right) \sin(\pi x)\;\drm x \\
  =& \frac{1}{\ell} \int_0^\ell \left[\cos\left(\frac{(n - 1)\pi}{\ell} x \right) - \cos\left(\frac{(n + 1)\pi}{\ell}x\right) \right]\; \drm x \\
\end{align*}
$$

Next, we integrate each cosine term separately. To solve the first integral, we substitute $r = \frac{(n - 1)\pi}{\ell}x$ and $\drm r = \frac{(n - 1)\pi}{\ell} \drm x$ to get

$$
\begin{align*}
  \int_0^\ell \cos\left(\frac{(n - 1)\pi}{\ell} x \right)\;\drm x =& \frac{\ell}{(n - 1)\pi} \int_0^{(n - 1)\pi} \cos(r)\; \drm u = \frac{\ell}{(n - 1) \pi} [\sin(r)]_0^{(n - 1)\pi} \\
  =& \frac{\ell\sin((n - 1)\pi)}{(n - 1)\pi}
\end{align*}
$$

Similarly, the second integral is solved as

$$
  \int_0^\ell \cos\left(\frac{\pi x (n + 1)}{\ell} \right)\;\drm x = \frac{\ell\sin((n + 1)\pi)}{(\ell + n)\pi}
$$

Combining the results we get

$$
\begin{align*}
  D_n = \ell \left(\frac{\sin((n - 1)\pi)}{(n - 1)\pi} - \frac{\sin((n + 1)\pi)}{(n + 1)\pi}\right) = \begin{cases} 1, \quad& n=1 \\ 0, \quad& n \geq 2 \end{cases}
\end{align*}
$$

This is expected as $\Set{\sin\left(\frac{n\pi}{\ell} x\right)}_{n\in\N}$ is an orthogonal basis for the Hilbert space $L^2 ([0,\ell])$. 

## Euler's Method

The notation and argumentation used to derive Euler's method is based on [@note_hjortjensen_2015, pp. 245-247]. Suppose we have a first order ordinary differential equation 

$$
  \frac{\drm }{\drm t} y(t) = y'(t) = f(t, y(t))
$$

with inital value $y(t_0) = y_0$. To apply Euler's method, we discretize the domain of $y$ with a fixed time step $h$. Taking the $n$th order Taylor expansion of $y$ at $(i+1)$th time step $t_i = ih + t_0$ gives

$$
\begin{align*}
  y(t_{i+1}) =& y(t_i + h) = y(t_i) + h\left(y'(t_i) +\cdots + y^{(p)}(t_i) \frac{h^{p-1}}{p!} \right) + \mathcal{O}(h^{p+1}) \\
  =& y_i + h\left(f(t_i, y(t_i)) +\cdots + f^{(p-1)}(t_i, y(t_i)) \frac{h^{p-1}}{p!} \right) + \mathcal{O}(h^{p+1}),
\end{align*}
$$

where $\mathcal{O}(h^{p+1})$ is the approximation error. Truncating at the first order gives the explicit (forward) Euler scheme

$$
  y(t_{i+1}) = y(t_i) + hf(t_i, y(t_i)) + \mathcal{O}(h^2).
$$

Each step produces an error of order $\mathcal{O}(h^2)$. With $n$ steps, the total error is of order $n\mathcal{O}(h^2) \approx \mathcal{O}(h)$ [@note_hjortjensen_2015, pg. 246].

## Source Code

### Explicit Forward Euler Scheme

```{code} python
:label: code:explicit-euler-scheme
:caption: Python implementation of the explicit forward Euler scheme for solving the one-dimensional heat equation.

import numpy as np

def heat_equation_forward_euler(
  initial_condition: Callable[[np.ndarray], np.ndarray],
  dx: float,
  dt: float | None = None,
  x_max: float = 1.0,
  t_max: float = 1.0,
  alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

  if dt is None:
    dt = dx**2 / (2 * alpha)

  x = np.arange(0, x_max + dx, dx)
  t = np.arange(0, t_max + dt, dt)

  nx = len(x)
  nt = len(t)

  rho = alpha * dt / (dx**2)
  print(f"Stability condition (should be <= 0.5): {rho}")
  if rho > 0.5:
    print("Warning: Numerical scheme may be unstable!")

  u_euler = np.zeros((len(x), len(t)))
  u_euler[:, 0] = initial_condition(x)

  for j in range(nt - 1):
    for i in range(1, nx - 1):
      u_euler[i, j + 1] = (
        rho * u_euler[i - 1, j]
        + (1 - 2 * rho) * u_euler[i, j]
        + rho * u_euler[i + 1, j]
      )

  return u_euler, x, t
```

### Feedforward Neural Network Solver

```{code} python
:label: code:neural-network-solver-1
:caption: Python implementation of a feedforward neural network (FFNN) solver for the one-dimensional heat equation. This shows the class structure the solver. The code relies on [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) for automatic differentiation and [Flax](https://flax.readthedocs.io/en/latest/) for building the FFNN.

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

class HeatEquationSolver:
  def __init__(
    self,
    initial_condition: Callable[[jnp.ndarray], jnp.ndarray],
    hidden_layers: list[int],
    activation_functions: list[Callable],
  ):
    self.initial_condition = initial_condition

    class FFNN(nn.Module):
      hidden_layers: list[int]
      activation_functions: list[Callable]

      @nn.compact
      def __call__(self, x):
        activations = self.activation_functions or [nn.relu] * len(self.hidden_layers)

        for neurons, a in zip(self.hidden_layers, activations):
          x = nn.Dense(
            features=neurons,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
          )(x)
          x = a(x)

        x = nn.Dense(1)(x)

        return x

    self.model = FFNN(hidden_layers, activation_functions)
    input_shape = (2,)
    self.key = jax.random.PRNGKey(0)
    self.params = self.model.init(self.key, jnp.zeros(input_shape))["params"]

  def ffnn_trial(self, params, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    ...

  def cost_function(self, params, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    ...

  def train_ffnn(self, spatial_size: int, time_size: int, 
    epochs: int = 1000, learning_rate: float = 0.001
  ) -> float:
    ...
```

```{code} python
:label: code:neural-network-solver-2
:caption: Python implementation of a feedforward neural network (FFNN) solver for the one-dimensional heat equation. The code relies on [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) for automatic differentiation and [Flax](https://flax.readthedocs.io/en/latest/) for building the FFNN.

  def ffnn_trial(self, params, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    x = jnp.atleast_2d(x)
    t = jnp.atleast_2d(t)

    x_t = jnp.concatenate([x, t], axis=1)

    prediction = self.model.apply({"params": params}, x_t)
    result = (1.0 - t) * self.initial_condition(x) + x * (1.0 - x) * t * prediction
    return result.squeeze()

  def cost_function(self, params, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    def u_func(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
      return self.ffnn_trial(params, x, t)

    du_dt = jax.jacrev(u_func, argnums=1)(x, t)
    d2u_dx2 = jax.jacrev(jax.jacfwd(u_func, argnums=0), argnums=0)(x, t)
    residual = du_dt - d2u_dx2

    return residual
```

```{code} python
:label: code:neural-network-solver-3
:caption: Python implementation of a feedforward neural network (FFNN) solver for the one-dimensional heat equation. The code relies on [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) for automatic differentiation and [Flax](https://flax.readthedocs.io/en/latest/) for building the FFNN.


  def train_ffnn(
    self,
    spatial_size: int,
    time_size: int,
    epochs: int = 1000,
    learning_rate: float = 0.001,
  ):
    x = jnp.linspace(0, 1, spatial_size)
    t = jnp.linspace(0, 1, time_size)

    X, T = jnp.meshgrid(x, t)

    x_train = X.reshape(-1, 1)
    t_train = T.reshape(-1, 1)

    def loss_fn(params):
      residuals = jax.vmap(lambda x, t: self.cost_function(params, x, t))(
        x_train, t_train
      )
      return jnp.mean(jnp.square(residuals))

    # Optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(self.params)

    # Update step
    @jax.jit
    def update(
      params: optax.Params, opt_state: optax.OptState
    ) -> tuple[optax.Params, optax.OptState, float]:
      loss, grads = jax.value_and_grad(loss_fn)(params)
      updates, new_opt_state = optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state, loss

    # Training loop
    for epoch in range(epochs):
      self.params, opt_state, loss = update(self.params, opt_state)

      if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
        print(f"Best loss: {best_loss}")

    return loss.item()
```