---
title: Project 1
authors:
  - name: Insert Name
site:
  template: article-theme
exports:
  - format: pdf
    template: plain_latex
    output: report.pdf
math:
  # Note the 'single quotes'
  '\R': '\mathbb{R}'
  '\Set': '{\left\{ #1 \right\}}'
bibliography: references.bib
abstract: |
  This project 
---

# Introduction

This project studies 

# Theory and Method

## Gradient Descent

The core problem in machine learning is to parameter estimation by minimizing a scalar cost function $C:\Theta\to\R$, where $\Theta$ denotes the parameter space. This optimization problem can be stated as

$$
  \hat{\boldsymbol{\theta}} \in \argmin_{\boldsymbol{\theta}\in\Theta} C(\boldsymbol{\theta})
$$

Gradient descent is a first-order optimization algorithm that iteratively updates the parameters in the direction of negative gradient of the cost function. Specifically, the update rule takes the form

$$
  \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \nabla_{\boldsymbol{\theta}_t} C(\boldsymbol{\theta})
$$

where $\eta_t$ is the *learning rate*. This method relies on the principle from multivariable calculus that the gradient points in the direction of the steepest ascent, which means that moving in the opposite direction (the negative gradient) will lead to a decrease in the cost function. 

## Feedforward Neural Network

An artificial neural network (ANN) is a computational graph model where the nodes, called neurons, are connected in a layered structure. The first and last layers are referred to as input and output layers, respectively, while the intermediate ones are called hidden layers. A feedforward neural network (FFNN) is a type of ANN in which the information only flows in one direction (see [](#mathematical-description-of-feedforward-neural-networks) for details).

Suppose we have an FFNN with $L\in\N$ hidden layers, each having $n_\ell$ neurons for $L\in\set{1,\dots,L}$. Assume further that the FFNN takes in an input $\mathbf{x}\in X\subseteq\mathbb{F}^P$ with $P$ features, and gives an output $\mathbb{y}\in Y\subseteq\mathbb{F}^Q$, where $\mathbb{F}$ is a field. In the case of using an FFNN as a regression model for the $2$-dimensional Franke function, the input space is $X = \R^2$, while the output space is $Y = \R$. The FFNN can be modelled as a function $\mathcal{F}: \mathbb{F}^{n_0}\to \mathbb{F}^{n_{L+1}}$ given by 

$$
  \mathcal{F} := \boldsymbol{\sigma}_{L+1} \circ \F_{L+1} \circ \boldsymbol{\sigma}_{L} \circ F_L \circ\cdots\circ \boldsymbol{\sigma}_1 \circ F_1,
$$

where $F_\ell :\mathbb{F}^{n_{\ell-1}} \to\mathbb{F}^{n_\ell}$ are affine transformations representing a forward pass from layer $\ell-1$ to $\ell$ and $\boldsymbol{\sigma}_\ell : \mathbb{F}^{n_\ell} \to \mathbb{F}^{n_\ell}$ are activation functions applied following a forward pass. In a forward pass from layer $\ell - 1$ to $\ell$, the affine transformation $F_\ell$ produces weighted pre-activations $\mathbf{z}^{(\ell)} \in \mathbb{F}^{n_\ell}$ given by

$$
  \mathbf{z}^{(\ell)} = F_i(\mathbf{a}^{(\ell - 1)}) = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell - 1)} + \mathbf{b}^{(\ell)},
$$

where
- $\mathbf{W}^{(\ell)} \in\mathbb{F}^{n_{\ell-1}\times n_{l}}$ is an $n_{\ell-1} \times n_{\ell}$ matrix representing linear weights,
- $\mathbf{b}^{(\ell)} \in\mathbb{F}^{\ell}$ is a bias vector
- $\mathbf{a}^{(\ell - 1)} = \boldsymbol{\sigma}_{\ell-1} (\mathbf{z}^{(\ell-1)})$ is the activation output from the previous layer $\ell - 1$.

### Backpropagation Algorithm

A feedforward neural network (FFNN) can be trained using stochastic gradient. To calculate the gradients for each layer iteratively, a technique called backpropagation can be used. Appendix [](#derivation-of-the-backpropagation-algorithm) outlines the derivation of the backpropagation algorithm, which can be explained in the following steps

1. **Calculate the output error:** Firs determine the output error $\boldsymbol{\delta}^{(L+1)}$ using the gradient of the cost with respect to the activations:

$$
  \boldsymbol{\delta}^{(\ell)} = \nabla_{\mathbf{a}^{(\ell)}} C_n \odot \sigma'_{\ell} (\mathbf{z}^{(\ell)}),
$$

where $\odot$ denotes the Hadamard (elementwise) product.

2. **Backpropagate the errors:** For each layer $\ell = L,L-1,\dots,1$ calculate the error $\boldsymbol{\delta}^{(\ell)}$ using

$$
  \boldsymbol{\delta}^{(\ell)} = (\mathbf{W}^{\ell + 1})^\top \boldsymbol{\delta}^{(\ell + 1)} \odot \boldsymbol{\sigma}'_{\ell} (\mathbf{z}^{(\ell)}).
$$

3. **Update weights and biases:** For each layer $\ell = L,L-1,\dots,1$ adjust the biases $\mathbf{b}^{(\ell)} = (b_j^{(ell)})_{j=1}^{n_\ell}$ and weights $\mathbf{W}^{(\ell)} = (w_{jk}^{(\ell)})_{j,k=1}^{n_{\ell-1}, n_{\ell}}$ according to the updates

$$
\begin{align*}
  \hat{w}_{jk}^{(\ell)} =& w_{jk}^{(\ell)} - \eta \delta_j^{(\ell)} a_k^{(\ell - 1)} \\
  \hat{b}_j^{(\ell)} =& b_j^{(\ell)} - \eta \delta_j^{(\ell)},
\end{align*}
$$

where $\eta$ is the learning rate.

### Activation Functions

Activation functions give artificial neural networks non-linear properties. Without them, a neural network is nothing more than a linear model, regardless of the depth. Additionally, activation functions restrict the outputs to specific ranges, which is essential depending on the application, such as for binary or multi-class classification tasks. 

This project studies the following activation functions (see Figure [](#figure-activation_functions)):
1. **Sigmoid function**: The sigmoid function $\sigma:\R\to (0,1)$, or the logistical function, is given by

$$
\begin{align*}
  \sigma(x) = \frac{1}{1 + e^{-x}} \\
  \frac{\d}{\d x} \sigma(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \sigma(x) (1 - \sigma(x))
\end{align*}
$$

This a an "S"-shaped function that output values between $0$ and $1$, and is thus suitable for representing probability. The sigmoid function suffers from the vanishing gradient problem [@notes_smets_2024, pp. 22-23]. When a sigmoid activation is close to $0$ or $1$, the gradient becomes very small due to its assymptotic behaviour. This leads to ineffective weight updates during backpropagation. 

2. **Rectified linear unit (ReLU)**: The ReLU function $\operatorname{ReLU}:\R \to [0,\infty)$ is given by

$$
\begin{align*}
  \operatorname{ReLU}(x) := \max\set{0,x} \\
  \frac{\d}{\d x}\operatorname{ReLU} = \begin{cases} 1,\quad x >& 0 \\ 0,\quad x <& 0 \end{cases}
\end{align*}
$$

The derivative of the ReLU behaves similarly to the Heaviside step function (although they are not technically identical). This property helps mitigate the vanishing gradient problem, as the derivative remains constant for positive activations, allowing gradients to propagate effectively during training. On the other hand, ReLU can lead "dying" neurons. When the input is negative, the derivative is zero, effectively deactivating the neuron. This inactivation causes the gradients of the upstream neurens to vanish [@notes_smets_2024, pp. 22-23]. 

1. **Leaky rectified linear unit (LRelu)**: The leaky ReLU function $\operatorname{LReLU}:\R\times\R \to\R$ is given by
   
$$
  \operatorname{LReLU}(x,\alpha) := \max\set{\alpha x, x} \\
  \frac{\d}{\d x}\operatorname{LReLU}(x, \alpha) = \begin{cases} 1,\quad x >& 0 \\ \alpha,\quad x <& 0 \end{cases}
$$

The leaky ReLU differs from the ordinary ReLU by introducing a small slope for negative values. This addresses the problem of "dying neurons" affecting the ReLU. This modification allows the function to maintain a small gradient for negative inputs, enabling the neuron to continue learning even when it receives negative activations.

```{figure} figures/activation_functions.pdf
:label: figure-activation_functions
:alt: Activation Functions.
:align: center

Plot of the sigmoid, rectified linear unit (ReLU) and Leaky ReLU activation functions.
```

# Results

## Regression with Feedforward Neural Network

This section presents the results of employing a feedforward neural network (FFNN) for regression tasks using synthetic data generated by the Franke function. We explored various activation functions in the hidden layers, including sigmoid, ReLU, and Leaky ReLU. The performance of the FFNN was compared to linear regression methods, specifically ordinary least squares (OLS) and ridge regression. We initialized the weights of the FFNN using a normal distribution scaled by a Xavier parameter, while the biases were set to a small constant value of $0.01$. The FFNN was configured with a constant learning rate and utilized a linear activation function for the output layer, appropriate for this regression problem.

### Hyperparameter Tuning with Random Search

#### Sigmoid Activation
Hyperparameter tuning was performed on a feedforward neural network (FFNN) to identify optimal layer architectures, learning rates, and $\ell_2$ regularization parameters. The tuning process employed a random search with the following parameters:

- Learning rate: $\eta \sim \operatorname{Uniform}(0.0001, 0.1)$
- Regularization parameter: $\lambda \sim \operatorname{Uniform}(1 \times 10^{-6}, 0.1)$
- Number of hidden layers: $L \sim \operatorname{Uniform}(1, 3)$
- Number of neurons (in powers of $2$): $2^{n_L}, n_L \sim \operatorname{Uniform}(2, 7)$

The FFNN models were trained on $1,000$ samples using $5$-fold cross-validation, with each model undergoing $100$ epochs of training and utilizing minibatches of size $32$.

Figure [](#figure-ffnn_regression_franke_hypertuning_architecture) illustrates the mean squared error for various hidden layer architectures resulting from the random search.  Overall, there are no clear trends in the performance of the FFNN models based on the number of hidden layers and neurons within each layer. This lack of discernible patterns may be partly attributed to the experimental design, as the chosen epoch size of $100$ may be insufficient for deeper FFNNs with an increasing number of neurons. Models with three hidden layers generally performed worse, suggesting a potential for overfitting; the added complexity of three layers may not be warranted for this regression task.

Table [](table-ffnn-regression-franke-hypertuning-architecture) lists the score metrics for the top five performing models. The best model featured hidden layers of [128, 8], with a learning rate of $\eta = \num{7.95e-02}$ and an $\ell_2$ regularization parameter of $\lambda = \num{4.42e-05}$. In general, the top-performing models across different depths exhibited learning rates on the order of $10^{-2}$ and regularization parameters in the range of $10^{-6}$ to $10^{-4}$

:::{table} Performance scores for the top $5$ feedforward neural network models obtained from a random search hyperparameter tuning. Here $\eta$ denotes the learning rate, and $\lambda$ denotes the $\ell_2$ regularization parameter.

:label: table-ffnn-regression-franke-hypertuning-architecture
:align: center

| Layers | η | λ | MSE | R^2 | Time [s] |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 128-8 | \num{7.95e-02} | \num{4.42e-05} | \num{1.37e-02} \pm \num{2.74e-03} | \num{8.35e-01} \pm \num{2.13e-02} | \num{5.25e-01} \pm \num{5.09e-02} |
| 64 | \num{9.76e-02} | \num{2.16e-05} | \num{1.51e-02} \pm \num{5.70e-03} | \num{8.18e-01} \pm \num{5.90e-02} | \num{1.78e-01} \pm \num{4.83e-03} |
| 16 | \num{9.77e-02} | \num{7.52e-06} | \num{1.69e-02} \pm \num{3.77e-03} | \num{7.95e-01} \pm \num{3.94e-02} | \num{7.17e-02} \pm \num{9.03e-03} |
| 64-32 | \num{8.60e-02} | \num{1.16e-04} | \num{1.98e-02} \pm \num{2.62e-03} | \num{7.57e-01} \pm \num{4.76e-02} | \num{3.06e-01} \pm \num{5.09e-03} |
| 16 | \num{8.33e-02} | \num{6.12e-06} | \num{2.01e-02} \pm \num{7.34e-03} | \num{7.59e-01} \pm \num{7.75e-02} | \num{1.03e-01} \pm \num{1.64e-02} |
:::


```{figure} figures/ffnn_regression_franke_hypertuning_architecture.pdf
:label: figure-ffnn_regression_franke_hypertuning_architecture
:alt: Mean Squared Error of Hidden Layer Architectures for Feedforward Network.
:align: center

Mean squared error for various hidden layer architectures of a feedforward neural network (FFNN) using sigmoid activation for hidden layers and linear activation for the output layer. The FFNN models were trained on $1,000$ samples using $5$-fold cross validation. The models were trained with $100$ epochs and minibatches of size $32$.
```

#### Comparison with Ordinary Least Squares and Ridge Regression

$[2,32,16,1]$

$\eta = 0.05$

$\lambda = \num{1e-5}$

:::{table} 

:label: table-ffnn-regression-franke-comparison
:align: center

| Model | MSE | $R^2$ | Time [s] |
| :-: | :-: | :-: | :-: |
| FFNN (sigmoid) | \num{4.15e-03} \pm \num{2.58e-03} | \num{9.51e-01} \pm \num{2.84e-02} | \num{1.71e+00} \pm \num{3.11e-01} |
| FFNN (ReLU) | \num{4.47e-04} \pm \num{1.48e-04} | \num{9.95e-01} \pm \num{1.49e-03} | \num{8.35e-01} \pm \num{1.42e-01} |
| FFNN (LReLU) | \num{3.95e-04} \pm \num{2.39e-04} | \num{9.95e-01} \pm \num{2.44e-03} | \num{8.10e-01} \pm \num{1.53e-01} |
| Flux FFNN (sigmoid) | \num{7.92e-02} \pm \num{1.30e-02} | \num{2.65e-02} \pm \num{3.83e-02} | \num{1.33e+01} \pm \num{7.60e+00} |
| Flux FFNN (ReLU) | \num{7.64e-02} \pm \num{7.91e-03} | \num{7.71e-02} \pm \num{3.26e-02} | \num{1.09e+01} \pm \num{3.46e-01} |
| Flux FFNN (LReLU) | \num{7.58e-02} \pm \num{1.20e-02} | \num{7.12e-02} \pm \num{4.19e-02} | \num{1.13e+01} \pm \num{4.91e-01} |
| OLS (p=14) | \num{1.63e-04} \pm \num{4.27e-05} | \num{9.98e-01} \pm \num{5.39e-04} | \num{1.53e-02} \pm \num{1.52e-03} |
| Ridge (p=15) | \num{1.61e-04} \pm \num{6.55e-05} | \num{9.98e-01} \pm \num{7.54e-04} | \num{1.52e-02} \pm \num{1.08e-03} |
:::


### Rectified Linear Unit Activation

### Leaky Rectified Linear Unit Activation

## Classification with Feedforward Neural Network

# Conclusion

# Appendix

## Code Repository

The Julia source code used to generate the test results is available at [https://github.com/semapheur/fys-stk4155](https://github.com/semapheur/fys-stk4155).

## Mathematical Description of Feedforward Neural Networks

An artificial neural, or a computing unit, can be modelled as an affine transformation followed by non-linear activation function. Specifically, if $\mathbf{x}\in X\subseteq\mathbb{F}^n$ is the input signal and $\mathbf{y}\in Y\subseteq\mathbb{F}^m$ is output signal, where $\mathbb{F}$ is a field, an artifical neuron takes the form:

$$
  \mathbf{y} = \sigma(\mathbf{Wx} + \mathbf{b}),
$$

where
- $\mathbf{W}\in\mathbb{F}^{m\times n}$ is an $m\times n$ matrix representing linear weights,
- $\mathbf{b}\in\mathbb{F}^m$ is a bias vector,
- $\sigma:\mathbb{F}^m \to Y$ is an activation function.

The input and output spaces can generally assume many forms depending on the application. Common forms for $Y$ include:
- the real line $\R$ for regression problems.
- binary values $\set{0,1}^n$ for classification problems.
- unit intervals $[0,1]^n$ for representing probabilities.

A feedforward neural network (FFNN) is a directed acyclic graph of artifical neurons, in which information only flows in one direction. The neurons of an FFNN are structured into a sequence of layers. In this structure, the first and last layers are called the input and output layers, respectively, while the intermediate ones are referred to as hidden layers. An FFNN with $L\in\N$ hidden layers can be modelled as a composition of affine transformations chained by activation functions. This construction can be formalized as follows. Let $n_0, n_1,\dots,n_{L+1}\in\N$ be the number of neurons in each layer and $\boldsymbol{\sigma}_1,\dots,\sigma_L$ be (vectorized) activation functions. Suppose $F_\ell :\^{n_{\ell-1}} \to\mathbb{F}^{n_\ell}$ are affine transformations given by

$$
  F_\ell (\mathbf{x}) = \mathbf{W}^{(\ell)} \mathbf{x} + \mathbf{b}^{(\ell)},
$$

where $\mathbf{W}^{(\ell)} \in\mathbb{F}^{n_\ell \times n_{\ell-1}}$ and $\mathbf{b}^{(\ell)} \in\mathbb{F}^{n_\ell}$ for $\ell\in\set{1,\dots,L+1}$. The FFNN is then represented by the composition $\mathcal{F}: \mathbb{F}^{n_0} \to \mathbb{F}^{n_{L+1}}$ given by

$$
  \mathcal{F} := \boldsymbol{\sigma}_{L+1} \circ \F_{L+1} \circ \boldsymbol{\sigma}_{L} \circ F_L \circ\cdots\circ \boldsymbol{\sigma}_1 \circ F_1.
$$

In a forward pass from layer $\ell - 1$ to $\ell$, the affine transformation $F_\ell$ produces weighted pre-activation $\mathbf{z}^{(\ell)} \in \mathbb{F}^{n_\ell}$ given by

$$
  \mathbf{z}^{(\ell)} = F_i(\mathbf{a}^{(\ell - 1)}) = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell - 1)} + \mathbf{b}^{(\ell)},
$$

where $\mathbf{a}^{(\ell - 1)} = \boldsymbol{\sigma}_{\ell-1} (\mathbf{z}^{(\ell-1)})$ is the activation output from the previous layer $\ell - 1$. For each layer $\ell\in{1,\dots,L+1}$, the pre-activation and activation for neuron $j$ is written

$$
\label{equation-neuron-activation}
\begin{split}
  z_j^{(\ell)} =& \sum_{i=1}^{n_{\ell-1}} w_{ji}^{(\ell)} a_i^{(\ell - 1)} + b_j^{(\ell)} \\
  a_j^{(\ell)} =& \sigma_\ell (z_j^{(\ell)}),
\end{split}
$$

where $w_{ji}^{(\ell)}$ is the weight from neuron $i$ in layer $\ell - 1$ to neuron $j$ in layer $\ell$, and $b_j^{(\ell)}$ is the bias for neuron $j$ in layer $\ell$.

### Derivation of the Backpropagation Algorithm

In this section we derive the backpropagation algorithm for training a feedforward neural netork (FFNN) using a quadratic cost function. Suppose have a training set $\set{(\mathbf{x}_n, \mathbf{y}_n) | \mathbf{x}\in\mathbb{F}^{n_0}, \mathbb{F}^{n_{L+1}}}_{n=1}^N$ of $N\in\N$ samples. We are interested in minimizing the mean squared error given by

$$
\label{equation-ffnn-cost-mse}
  C(\mathbf{W}, \mathbf{b}) = \frac{1}{2N} \sum_{n=1}^N \lVert \hat{\mathbf{y}}_n - \mathbf{y} \rVert^2,
$$

where $\hat{\mathbf{y}} = \mathbf{a}^{(L+1)} = \sigma(\mathbf{z}^{(L+1)})$ is the output of FFNN for the $n$th input. The cost of training example $n$, given by

$$
\label{equation-ffnn-cost-mse-example}
  C_n (\mathbf{W}, \mathbf{b}) = \frac{1}{2} \lVert \mathbf{a}^{(L+1)} - \mathbf{y}_n \rVert^2 = \frac{1}{2} \sum_{j=1}^{n_{L+1}} (a_j^{(L+1)} - y_n)^2
$$

To derive the backpropagation equations, we need to calculate the partial derivatives of $C_n$ with respect to the bias and the weights. From [](#equation-neuron-activation), we obtain the partial derivatives

$$
  \frac{\partial z_j^{(\ell)}}{\partial w_{kj}^{(\ell)}} =& a_k^{(\ell - 1)},\quad
  \frac{\partial z_j^{(\ell)}}{\partial b_j^{(\ell)}} =& 1,\quad
  \frac{\partial a_j^{(\ell)}}{\partial z_j^{(\ell)}} =& \sigma'_{\ell}(\mathbf{z}_j^{\ell})
$$

We also define the error $\delta_j^{(\ell)}$ for neuron $j$ in layer $\ell$ by 

$$
  \delta_j^{(\ell)} := \frac{\partial C_n}{\partial z_j^{(\ell)}} = \frac{\partial a_j^{(\ell)}}{\partial z_j^{(\ell)}} \frac{\partial C_n}{\partial a_j^{(\ell)}} = \sigma'_\ell (z_j^{(\ell)}) \frac{\partial C_n}{\partial a_j^{(\ell)}} 
$$

In matrix form, this can be written as

$$
  \boldsymbol{\delta}^{(\ell)} = \nabla_{\mathbf{a}^{(\ell)}} C_n \odot \sigma'_{\ell} (\mathbf{z}^{(\ell)})
$$

where $\odot$ denotes the Hadamard (elementwise) product. For the output layer $\ell = L + 1$ we find from [](#equation-ffnn-cost-mse), that $\nabla_{\mathbf{a}^{(L+1)}} C_n = \mathbf{a}^{(L)} - \mathbf{y}_n$ leading to the output error

$$
  \boldsymbol{\delta}^{(L)} = (\mathbf{a}^{(L)} - \mathbf{y}_n) \odot \sigma'_{L+1} (\mathbf{z}^{(L)})
$$

For any hidden layer $\ell < L + 1$, the partial derivative of $C_n$ with respect to an activation $a_k^{(\ell)}$ is found by summing up the errors $\delta_j^{(\ell + 1)}$ of the next layer $\ell + 1$, i.e.,

$$
\label{equation-neuron-error-hidden}
  \frac{\partial C_n}{\partial a_k^{(\ell)}} = \sum_{j=1}^{n_{\ell+1}} \underbrace{\frac{\partial z_j^{(\ell + 1)}}{\partial a_k^{(\ell)}}}_{=w_{kj}^{(\ell+1)}} \underbrace{\frac{\partial a_j^{(\ell + 1)}}{\partial z_j^{(\ell+1)}} \frac{\partial C_n}{\partial a_j^{(\ell+1)}}}_{\delta_j^{(\ell+1)}} = \sum_{j=1}^{n_{\ell+1}} w_{kj}^{(\ell + 1)} \delta_{j}^{(\ell+1)}
$$

In matrix form this can be expressed as $\nabla_{\mathbf{a}^{(\ell)}} C_n = (\mathbf{W}^{(\ell+1)})^\top \boldsymbol{\delta}^{(\ell+1)}$. Thus, for any hidden layer $\ell < L + 1$, the error $\boldsymbol{\delta}^{(\ell)}$ can be expressed as

$$
  \boldsymbol{\delta}^{(\ell)} = (\mathbf{W}^{\ell + 1})^\top \boldsymbol{\delta}^{(\ell + 1)} \odot \boldsymbol{\sigma}'_{\ell} (\mathbf{z}^{(\ell)}).
$$

#### Gradient Computation

It remains to calculate the partial derivatives of the cost $C_n$ with respect to the bias and the weights. For any layer $\ell$, the partial derivative of $C_n$ with respect to a weight $w_{kj}^{(\ell)}$ is given by the chain rule

$$
  \frac{\partial C_n}{\partial w_{kj}^{(\ell)}} = \frac{\partial z_j^{(\ell)}}{\partial w_{kj}^{(\ell)}} \underbrace{\frac{\partial a_j^{(\ell)}}{\partial z_j^{(\ell)}} \frac{\partial C_n}{\partial a_j^{\ell}}}_{=\delta_j^{(\ell)}} = a_k^{(\ell-1)} \delta_j^{(\ell)}
$$

Likewise, the partial derivative of $C_n$ with respect a bias $b_j^{(\ell)}$ is given by

$$
  \frac{\partial C_n}{\partial b_j^{(\ell)}} = \frac{\partial z_j^{(\ell)}}{\partial b_j^{(\ell)}} \underbrace{\frac{\partial a_j^{(\ell)}}{\partial z_j^{(\ell)}} \frac{\partial C_n}{\partial a_j^{\ell}}}_{=\delta_j^{(\ell)}} = \delta_j^{(\ell)}
$$

The backpropagation algorithm can be summed up as follows.

:::{prf:algorithm} Backpropagation Algorithm
:label: algorithm-backpropagation

1. **Find the output error:** Calculate the output error $\boldsymbol{\delta}^{(L+1)}$ using the gradient of the cost with respect to the activations:

$$
  \boldsymbol{\delta}^{(\ell)} = \nabla_{\mathbf{a}^{(\ell)}} C_n \odot \sigma'_{\ell} (\mathbf{z}^{(\ell)}).
$$

2. **Backpropagate the errors:** For each layer $\ell = L,L-1,\dots,1$ calculate the error $\boldsymbol{\delta}^{(\ell)}$ using

$$
  \boldsymbol{\delta}^{(\ell)} = (\mathbf{W}^{\ell + 1})^\top \boldsymbol{\delta}^{(\ell + 1)} \odot \boldsymbol{\sigma}'_{\ell} (\mathbf{z}^{(\ell)}).
$$

3. **Update weights and biases:** For each layer $\ell = L,L-1,\dots,1$ adjust the biases $\mathbf{b}^{(\ell)}$ and weights $\mathbf{W}^{(\ell)}$ according to the updates

$$
\begin{align*}
  \hat{w}_{jk}^{(\ell)} =& w_{jk}^{(\ell)} - \eta \delta_j^{(\ell)} a_k^{(\ell - 1)} \\
  \hat{b}_j^{(\ell)} =& b_j^{(\ell)} - \eta \delta_j^{(\ell)},
\end{align*}
$$

where $\eta$ is the learning rate.
:::

## Franke's Function

In this project, Franke's function was used to generate training data for the polynomial regression models. This is a two-dimensional scalar field $f:\R^2 \to\R$ given by a weighted sum of four exponentials:

$$
\label{equation-12}
\begin{split}
  f(x,y) =& \frac{3}{4} \exp\left(-\frac{(9x - 2)^2}{4} - \frac{(9y - 2)^2}{4} \right) \\
  &+ \frac{3}{4}\exp\left(\frac{(9x + 1)^2}{49} - \frac{9y + 1}{10}\right) \\
  &+ \frac{1}{2}\exp\left(-\frac{(9x - 7)^2}{4} - \frac{(9y - 3)^2}{4} \right) \\
  &- \frac{1}{5}\exp\left(-(9x - 4)^2 - (9y - 7)^2 \right)
\end{split}
$$

A plot of the Franke function on the unit square $[0,1]^2$ is given in [](#figure-franke).

```{figure} figures/franke.svg
:label: figure-franke
:alt: Franke function
:align: center

Plot of the Franke function on the unit square $[0,1]^2$.
```

## Source Code

### Feedforward Neural Network


```{code} julia
:label: code-ffnn
:caption: Neural network implementation in Julia.

"""
Layer(weights, biases, activation, activation_derivative)

A struct representing a layer in a neural network.

# Fields
- `weights::Matrix{Float64}`: A matrix of weights for the layer.
- `biases::Vector{Float64}`: A vector of biases for the layer.
- `activation::Function`: The activation function for the layer.
- `activation_prime::Function`: The derivative of the activation function for the layer.
"""
struct Layer
  weights::Matrix{Float64}
  biases::Vector{Float64}
  activation::Function
  activation_prime::Function
end

"""
NeuralNetwork

A struct representing a neural network.

# Fields
- `layers::Vector{Layer}`: A vector of layers in the network.
- `cost::Function`: The cost function used to evaluate the network.
- `cost_prime::Function`: The derivative of the cost function used to evaluate the network.
- `lr_scheduler::LearningRateScheduler`: The learning rate scheduler used to update the network.
- `l2_lambda::Float64`: The regularization strength for L2 regularization.
"""
struct NeuralNetwork
  layers::Vector{Layer}
  cost::Function
  cost_prime::Function
  lr_scheduler::LearningRateScheduler
  l2_lambda::Float64
end
```