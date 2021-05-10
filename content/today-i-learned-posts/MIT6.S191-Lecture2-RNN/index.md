---
title: MIT 6.S191 - Lecture 2 - Recurrent Neural Networks
author: ''
date: '2021-02-19'
categories: [course, python, deep learning]
tags: [course, python, deep learning]
summary: 'Personal notes of `MIT 6.S191`, Lecture 2: Recurrent Neural Networks'
reading_time: yes
image:
  caption: ''
  focal_point: ''
  preview_only: true
type: today-i-learned
draft: false
---

# <span style="color:#FF9F1D"> Intro </span>

From a single perceptron, we can extend the number of inputs, neurons and hield multi-dimensional outputs:

{{< figure src="images/L2_multi_output.png" title="" lightbox="true" >}}

But this multi perceptron, or Neural Network, doesn't have a sense of time or element sequence. Every input and output is a specific time step.

{{< figure src="images/L2_sequence.png" title="Lack of connection between Multi-dimensional perceptrons input in sequences" lightbox="true" >}}

This lack of connection between time steps is problematic in predicting problems that involves time or sequences. In a sequence, the inputs are correlated with each other. They are not independent. For example, future sales in a given shop are correlated with previuos sales, they are not independent events.

Expresing it in the above graph, the output of $\hat{y}_2$ not only depends on $X_2$, but also on $X_0$ and $X_1$.

# <span style="color:#FF9F1D"> The missing piece, the Cell state </span>

To make use of the correlation of the inputs in sequence, the network would need to have a connection that allows to look forward**. This connection is called internal memory or **cell state** $h_t$:

{{< figure src="images/L2_rnn.png" title="Using a memory cell to pass throw previous input information" lightbox="true" >}}

**The memory or cell state pass the current information in the step $t$ to the next step $t+1$**.


# <span style="color:#FF9F1D"> Recurrent Neural Networks </span>

Recurrent Neural Networks are the result of incorporating the idea of using cell states to pass throw information between time steps. **They can be thought of as multiple copies of the same network, each passing the new cell state value to a successor network**. Every network is a time step of the *global* neural network.

RNNs have a state $h_t$, that is updated at each time step as a sequence is processed. The recurrent relation applied at each and every time step is defined as:

{{< figure src="images/L2_rec_rel.png" title="Cell state as a weighted function of current inputs and previous cell states" lightbox="true" >}}

The function is going to be parametrized by a set of weights that is leaned throughout training the model. **The same function and the very same parameters are applied every step of processing the sequence (every iteration of the model)**.

{{< figure src="images/L2_rnn_ac.png" title="Pass throw of THE SAME weigthed matrix as input for the new state" lightbox="true" >}}

- $W_{xh}$ denotes the weight matrix optimized for that specific step of the sequence.

- $W_{hh}$ denotes the weight matrix of the memory cell, reused every step for the entire sequence.

- $W_{hy}$ denotes the weight matrix of a combination of both the specific optimization of the weights for that step, and the memory cell matrix.

In practice, you won't see the cell states weigthing the outputs of the next step outputs, or multiple networks one after the other. The loop is made inside one single architecture. The RNN algorithm can be simplyfied as:

{{< figure src="images/L2_rnn_eq.png" title="Output of the state cell as a function of the weighted hidden memory cell and current inputs" lightbox="true" >}}


# <span style="color:#FF9F1D"> Examples of RNN application </span>

Recurrent Neural Networks are usually used in text problems such as sentiment classification, text generation from an image, generation of image title or translation.

{{< figure src="images/L2_cat_words.png" title="" lightbox="true" >}}

This is an example using **many** words **to predict the one** next word in the sentence. Depending on the problem, the number of inputs and outputs change, that modify the NN architecture:

{{< figure src="images/L2_examples_rnn.png" title="Different examples of RNNs with varying inputs and outputs" lightbox="true" >}}

# <span style="color:#FF9F1D"> Making Neural Networks understand text: Embedding </span>

Neural Networks do not understand word language, or images, they only understand numbers. They require the words to be parsed as vectors or arrays of numbers:

{{< figure src="images/L2_words.png" title="Words to Vectors" lightbox="true" >}}

**How are this vectors made?**

1. The computer/algorithm gets all the words and create a **vocabulary** with them.

2. Then, it creates its own dictionary to understand them, assigning a number to each different word (**indexing**).

3. The numbers form vectors of a fixed size that captures the content of the word (**embedding**).

By using vectors and not single numbers, you can compare how close are vectors to each other. And comparing distance is key because the words that usually go together in a phase must be represented by vectors close to each other. For example, the vector of *dog* is closer to the vector of *cat* than to the vector of *sad*.

**Embedding gather words together by similarity using the distance between vectors.**


{{< figure src="images/L2_embedding.png" title="Embedding or vectorizing words" lightbox="true" >}}

# <span style="color:#FF9F1D"> Model Design Criteria, or why RNN are good </span>

Any recurrent model architecture must the following design criteria:

1. Must handle variable-length sequences (RNN ✔️)

{{< figure src="images/L2_length.png" title="RNN must handle different length phrases" lightbox="true" >}}

2. Must track long-term dependencies (RNN ✔️)

{{< figure src="images/L2_long_dep.png" title="RNN must keep track of long-term inputs" lightbox="true" >}}

3. Must mantain information about order (RNN ✔️)

{{< figure src="images/L2_order.png" title="RNN must keep track of inputs order" lightbox="true" >}}

4. Must share parameters across the sequence (RNN ✔️)

In RNNs the same memory cell is reused every step for the entire sequence, as explained previusly.


# <span style="color:#FF9F1D"> RNN Ilustrated example (from Michael Phi) </span>

Let's say that we want to do a many-to-one prediction in which the inputs are words in this cereal review and the output is a positive or negative sentiment analysis.

![](https://miro.medium.com/max/1400/1*YHjfAgozQaghcsEvsBEu2g.png)

First the words are transformed to vectors by embedding.

From:

{{< figure src="images/L2_LSTM_1.png" title=" " lightbox="true" >}}

TO:

![](https://miro.medium.com/max/1400/1*AQ52bwW55GsJt6HTxPDuMA.gif)

While processing, it passes the previous hidden state to the next step of the sequence. The hidden state acts as the neural networks memory. It holds information on previous data the network has seen before.

![](https://miro.medium.com/max/1400/1*o-Cq5U8-tfa1_ve2Pf3nfg.gif)

For every of theses steps or layers, the input and previous hidden state are combined to form a vector. It goes through a tanh activation, and the output is the new hidden state $h_t$. The tanh function ensures that the values stay between -1 and 1.

![](https://miro.medium.com/max/1400/1*WMnFSJHzOloFlJHU6fVN-g.gif)


# <span style="color:#FF9F1D"> Backpropagation Through Time (BPTT) </span>

The usual NN backpropagation algorithm:

1. Take the derivative (gradient) of the loss with respect to each parameter $W$.
2. Shift parameters to minimize loss.

With a basic Neural Network, the backpropagation errors goes trough a single feedforward network for a single time step.

Recurrent Network backpropagation needs a twist, as it contains multiple steps and a memory cell. In RNNs, **the errors are backpropagating from the overall loss through each time step**:

{{< figure src="images/L2_BPTT.png" title="RNN backpropagation visualization" lightbox="true" >}}

The key difference is that the gradients for $W$ at each time step are summed. A traditional NN doesn't share parameters across layers. Every input is different and have different weigths $W$.

# <span style="color:#FF9F1D"> Problems with backpropagation in RNN </span>

Computing the gradient with respect to the initial $h_0$ involves many matrix multiplications between the memory cell $h_t$ and the weights $W_hh$.

### Exploiting gradients (gradients > 1.0)

In the the process of backpropagation the gradients get multiplied by each other over and over again. If they are larger than 1.0, the end matrix of weigths is huge.

As a silly example: 0.5 times 1.5 is 0.75, 0.5 times 1.5^200 is 8.2645996e34. This can give you a perspective of how matrix multiplication can explote by mutliplying constantly by 1.X.

These huge gradients can become extremely large as the result of matrix and the loss function cannot be minimized.

The usual solution is change the derivative of the errors before they propagate through the network, so they don't become huge. Basically, you can create a threshold that the gradients cannot surpass. *Create a threshold* means that you set a value, such as 1.0, that forces the values to be 1.0 at maximum.


### Avoid exploiting gradients: Gradient thresholds

There are two ways to create these thresholds:

**1. Gradient Norm Scaling**

Gradient norm scaling rescales the matrix so the gradient equals 1.0 if the a gradient exceeds 1.0.

{{% alert look%}}
Gradient Norm Scaling in Tensorflow:
{{% /alert %}}

```python
  opt = SGD(lr=0.01, clipnorm=1.0)
```

**2. Gradient Value Clipping**

Gradient value clipping simply forces all the values above the threshold to be the threshold, without changing the matrix. If the clip is 0.5, all the gradient values less than -0.5 are set to -0.5 and all the gradients more than 0.5 set to 0.5.

{{% alert look%}}
Gradient Norm Scaling in Tensorflow:
{{% /alert %}}

```python
  opt = SGD(lr=0.01, clipvalue=0.5)
```


### Vanishing gradients (gradients < 1)

As gradients can become huge they can also become tiny to the point that it is not possible to effectively train the network.

This is a problem because the errors further back in time are not being propagated. It would cause that the long-term errors are vanished and bias the model only to capture short-term dependencies.

### Avoid vanishing gradients

The basic recipe to solve vanishing gradients is use a ReLU activation function, chaning to a smart weight initialization and/or use a different RNN architecture.

**1. Change activation function to ReLU.**

Why ReLu?

Because when the cell or instance gets activated (weight 0 or more), by definition the derivative or gradient is 1.0 or more:

{{< figure src="images/L2_activation_trick.png" title="Solution 1: Use a ReLU activation function" lightbox="true" >}}

**2. Change weight initialization.**

For example to the **Xavier initialization/Glorot initialization**:

{{% alert look%}}
Changing the weight activation in Tensorflow:
{{% /alert %}}

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(16, input_shape=(1,5), activation='relu'),
    Dense(32, activation='relu', kernel_initializer='glorot_uniform'),
    Dense(2, activation='softmax')
])
```
**3. Change Network architecture.**

More complex RNNs such as **LSTM or GRU** can control the information that is passing through. Long Short Term Memory networks (**LSTM**) and Gated Recurrent Units (**GRU**) are special kinds of RNN, capable of learning long-term dependencies.

{{< figure src="images/L2_activation_trick3.png" title="Solution 3: Use a NN with fated cells" lightbox="true" >}}

They can keep informed of long-term dependencies **using filters or gates**. In essence, these gates decide how much information to keep of the previous neuron state or values, and how much to drop. This makes the optimization problem or the Neural Network less prompt to vanishing or exploding gradient problems.

# <span style="color:#FF9F1D"> LSTM </span>

In a simple RNN, the information goes though every step with the input of that time step ($x_t$), the previous step memory cell ($h_{t-1}$) and an output for every step ($y_t$).

{{< figure src="images/L2_rnn_arq.png" title="RNN arquitecture vizualization" lightbox="true" >}}

The structure of a LSTM is more complex. **LSTM forces the matrix inputs in every step to go through gates**, or internal mechanism to keep long-term information.

{{< figure src="images/L2_lstm_arq.png" title="LSTM arquitecture vizualization" lightbox="true" >}}

# <span style="color:#FF9F1D"> LSTM Gates system

They 4 types of gates interacting within each step layer:

1. ***Forget gate***: Remove the irrelevant information.

Information from the previous hidden state and the current input is passed through the sigmoid function. Values come out between 0 and 1.

The closer to 0 means to forget, and the closer to 1 means to keep.

![](https://miro.medium.com/max/1400/1*GjehOa513_BgpDDP6Vkw2Q.gif)

2. ***Store gate***: Store relevant information.

The same previous $h_{t-1}$ and the current inputs goes into two transformations:

- Sigmoid transformation. It is the same operation as before, but in another gate. Instead of forget and keep, it will decide the information to update or not update.

- Than transformation. It helps to regulate the network by squishing values between -1.0 and 1.0.

The matrix multiplication of the tanh outputs with the sigmoid outputs decides which information is important, and store it in a cell state $\bigotimes$.

![](https://miro.medium.com/max/1400/1*TTmYy7Sy8uUXxUXfzmoKbA.gif)

3. ***Update gate***: update the separated cell state.

- The update gate takes the previous cell state vector $c_{t-1}$ and multiply by the forget vector (from the forget gate), that allows to drop non-important information.

- Then, it adds the store vector from the store gate, as this information is important to keep from the current step.

![](https://miro.medium.com/max/1400/1*S0rXIeO_VoUVOyrYHckUWg.gif)

The update gate takes the information to the other 2 gates to decide what to forget and what to keep, updating the cell state.

4. ***Output gate***: decides what the next hidden state $h_{t+1}$.

- The previous hidden state and the current input into a sigmoid function.
- Then the newly modified cell state pass the tanh function.
- By multiplying the two vectors it decides what information the hidden state should carry.

![](https://miro.medium.com/max/1400/1*VOXRGhOShoWWks6ouoDN3Q.gif)


# <span style="color:#FF9F1D"> GRU </span>

GRU’s has fewer tensor operations; therefore, they are a little speedier to train then LSTM’s. There isn’t a clear winner which one is better, try both to determine which one works better for their use case.

![](https://miro.medium.com/max/1400/1*jhi5uOm9PvZfmxvfaCektw.png)
