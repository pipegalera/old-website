---
title: MIT 6.S191 - Lecture 1 - Neural Networks
author: ''
date: '2021-02-11'
categories: [course, python, deep learning]
tags: [course, python, deep learning]
summary: 'Personal notes of `MIT 6.S191`, Lecture 1: Intro to Deep Learning'
reading_time: yes
image:
  caption: ''
  focal_point: ''
  preview_only: true
type: today-i-learned
draft: false
---

# <span style="color:#FF9F1D"> Perceptron </span>


{{< figure src="images/perceptron.png" title="" lightbox="true" >}}

If we denote $\hat{y}$ as the output:


$$\begin{array}{c}
\hat{y}=g\left(w_{0}+\sum_{i=1}^{m} x_{i} w_{i}\right)
\end{array}$$


Being $g$ , for example, a Sigmoid, Tangent or ReLU function:


$$
g(z)=\frac{1}{1+e^{-z}} \quad , \quad g(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}} \quad , \quad g(z)=\max (0, z)
$$


The purpose of activation functions is to introduce non-linearity into the network:

{{< figure src="images/linear.png" title="Linear vs Non-linear classification boundary" lightbox="true" >}}


Linear activation functions produce linear decisions no matter the network size while non-linearities allow approximating arbitrarily complex functions.


# <span style="color:#FF9F1D"> Neural Networks </span>

Taking the previous perceptron and simplifying the output to be $z$:

{{< figure src="images/image2.png" title="" lightbox="true" >}}

We can try with different weights, that would produce different outputs $z_1$ and $z_2$:

{{< figure src="images/image3.png" title="" lightbox="true" >}}

Neural Network is made stacking those different outputs. Notice that this is just a stack of dot products of the same features and different weights ($W^{(1)}$).

These outputs in the hidden layer have a different range of values, but there are only 2 possible final outputs: $\hat{y_1}$ and $\hat{y_2}$.

**How we classify a label as $\hat{y_1}$ or $\hat{y_2}$.?**

In this step the non-linear or transformation function $g$ trigger the outcomes to being one or the other.

- If the outcome value is more than the function threshold, the outcome is transformed to 1 (the label of $\hat{y_1}$).

- If the value is less than the threshold, the outcome is transformed to 0 (the label of $\hat{y_2}$).

{{< figure src="images/L1_network.png" title="Network drawing" lightbox="true" >}}

{{% alert look%}}
Neural Network application in Tensorflow:
{{% /alert %}}


```python
import tensorflow as tf

model = tf.keras.Sequential([
        # Hidden layers with n neurons
        tf.keras.layers.Dense(n),
        # Output layer with 2 neurons
        tf.keras.layers.Dense(2)
])
```

*Dense* means that the layers are fully connected, all the neuron's weight counts in the dot product calculation.

# <span style="color:#FF9F1D"> Forward propagation in Matrix notation (extra explanation) </span>

For example, let's say that we have 3 observations, we know 2 features of them, and we want to construct a Neural Network with 1 hidden layer containing 3 neurons.


- In a first step (1), we calculate manually the dot product of $X$ and $W^{(1)}$:

$$Z = XW^{1}$$

**The shape of $Z$ is always a product of: *(observations, features) x (features, n neurons in the layer)***.

The columns of the first element have to be equal to the rows of the second element. It is necessary for matrix calculation.

- The second step (2), we take the outputs of the hidden layer, apply the non-linear transformation, and calculate the dot product with respect to the second layer of weights:

$$\hat{y} = g(Z)W^{2}$$


Here is an example of how to calculate $\hat{y}$ using the dot product for a made-up dataset:

{{< figure src="images/L1_matrix.jpg" title="Forward propagation matrix calculation example" lightbox="true" >}}

The final output is 3 predictions (*real numbers*) for the 3 observations. Imagine that all the notations denoted with $w$ are constants chosen randomly. Then, every matrix product is also constants as the only variable that is an incognita are these weights.

Weight updating is made by the network by backward propagation (later explained).

# <span style="color:#FF9F1D"> Deep Neural Networks </span>

To make a Neural Network deep, we just add more layers. The number of layers and the number of neurons of each layer has to be defined beforehand (parameters to optimize) by us, humans. The model is only tunning the weights.

{{% alert look%}}
Neural Network application in Tensorflow:
{{% /alert %}}

```python
import tensorflow as tf

model = tf.keras.Sequential([
        # Hidden layers with n neurons
        tf.keras.layers.Dense(n),
        # Hidden layers with n neurons
        tf.keras.layers.Dense(n),
        # Output layer with 2 neurons
        tf.keras.layers.Dense(2)
])
```

# <span style="color:#FF9F1D"> The loss function </span>

Initiating random values of $W$, will give a prediction. A terrible one, as the model has no idea yet if the prediction is good, or how to measure how good is it.

**The measure of how good is a prediction is will be determined by the *Loss function***.

The "Loss function" measures how bad is the prediction. The final output predictions compares the predicted values with the actual ones:

$$
\mathcal{L}\left(f\left(x^{(i)} ; \boldsymbol{W}\right), y^{(i)}\right)
$$

The more the difference, the worse the prediction as predicted values are far away from the real ones. We want to minimize the loss function.

On average, for all the $n$ observations:

$$
\boldsymbol{J}(\boldsymbol{W})=\frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\left(f\left(x^{(i)} ; \boldsymbol{W}\right), y^{(i)}\right)
$$

# <span style="color:#FF9F1D"> Training the Neural Network: Gradient Descent and Backpropagation </span>

The final goal of every Neural Network is find the weights that achieve the lowest loss:

$$
\boldsymbol{W}^{*}=\underset{\boldsymbol{W}}{\operatorname{argmin}} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\left(f\left(x^{(i)} ; \boldsymbol{W}\right), y^{(i)}\right) $$
$$
\boldsymbol{W}^{*}=\underset{\boldsymbol{W}}{\operatorname{argmin}} J(\boldsymbol{W})
$$
**How the Neural Network finds the optimal ${W}^{*}$? By gradient descent.**

Gradient descent algorithm:

1. Initialize wrights randomly.
2. Compute the gradient.
3. Update the weights according to the direction of the gradient and the learning rate.
4. Loop until convergence 2 and 3.
5. Return optimal weights.


# <span style="color:#FF9F1D"> Backpropagation </span>

In the second step, the algorithm computes the gradient by a process called backpropagation. **Backpropagation is just the efficient application of the chain rule** for finding the derivative of the loss function with respect to the neuron weights.

{{< figure src="images/backpropagation.png" title="Backpropagation as an application of the chain rule" lightbox="true" >}}

When training a neural net, the goal is to find neuron parameters (weights) that cause the output of the NN to best fit the data, right? The chain rule is the way the NN can "connect" the loss function and outputs with the weight parametrization.

- If the loss function is less than the previous value using the current weights, then the gradient is in a good direction.

- If the loss function is more than the previous, it goes in the opposite direction.

- Repeat until the loss function is zero or cannot make it lower (*convergence*).

When the Neural Network converged, it found a spot in the loss function that increasing or decreasing the weight values makes the loss function increasing.

Note that it might be the case that the optimal weights are not optimal for the entire loss function space because they converged in a local minimum. In practice, finding the global minimum is very difficult as the algorithm is very prompt to get stuck in these local minimums along the way of convergence.

{{< figure src="images/gradient_landscape.png" title="Visualizing the loss landscape of Neural Nets" lightbox="true" >}}


# <span style="color:#FF9F1D"> Learning rates </span>

**The learning rate is how much increase the weight in the updating step of the gradient descent.**. If the gradient calculates the direction of the algorithm to find the minimum, the learning rate sets the magnitude of every weight try.

Setting a stable learning rate is key to find the global minimums. It should be large enough that avoid local minimums, but small enough that is not being able to convergence (**Exploding Gradient Problem or Divergence**). Stable learning rates converge smoothly and avoid local minima.

In practice, a usual approach is trying a lot of different learning rates and see what works. A better one is to design an adaptative learning rate that "adapts" to the loss function or landscape. In this second approach, the learning rate is no longer a constant or fixed number but a rate that gets smaller or bigger depending on how large the gradient is, how fast the learning is happening, the size of the particular weights, and so forth.

In Tensorflow, these are called optimizers. They are many learning rate optimizers that make the NN coverage more quickly and generally better such as Adaptive Gradient Algorithm (Adam) or Adadelta.

{{% alert look%}}
Optimizers application in Tensorflow:
{{% /alert %}}

```python
tf.keras.optimizers.Adam
tf.keras.optimizers.Adadelta
```

# <span style="color:#FF9F1D"> Batching and Stochastic gradient descent </span>

When we talked about backpropagation and computing the gradient, I did not mention how computationally expensive this can be. In practice, calculating the chain rule for hundreds of layers using the entire training set every time the algorithm loops is not feasible.

**Instead of looping through the entire training set, we can pick a random sub-sample of the data. This process is also called *Batching*** as it divides the training sets into small batches of data that feed the NN. The gradient computation is passed only through a small batch of data $B$:
$$
\frac{\partial J(W)}{\partial W}=\frac{1}{B} \sum_{k=1}^{B} \frac{\partial J_{k}(W)}{\partial W}
$$

Then the weights are updated accordingly and the process starts again with another sub-sample or batch.

**This process is called *Stochastic gradient descent*, as it replaces the actual gradient (calculated from the entire data set) by an estimate thereof (calculated from a randomly selected subset of the data).**

# <span style="color:#FF9F1D"> Regularization </span>

A technique that **constrains the optimization problem** to discourage complex models to avoid overfitting.

### Regularization I: Dropout

**For every iteration, the Neural Network drops a percentage of the neurons.**

Using Dropout the Neural Network doesn't rely on a pathway or very heavy weighting on certain features and overfitting, making the Neural Network more prompt to generalize to new data.

{{< figure src="images/dropout.png" title="Dropout regularization" lightbox="true" >}}


{{% alert look%}}
Dropout regularization in Tensorflow:
{{% /alert %}}

```python
tf.keras.layers.Dropout(p=0.5)
```

### Regularization II: Early stopping

First, we monitor the process of minimizing the loss function of training and testing data at the same time.

When the loss function starts increasing in the test data (more difference between predicted and real outputs), stop the Neural Network.

{{< figure src="images/early_stopping.png" title="Early stoping regularization" lightbox="true" >}}
