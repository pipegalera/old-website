---
title: MIT 6.S191 - Lecture 3 - Convolutional Neural Networks
author: ''
date: '2021-04-06'
categories: [course, deep learning, computer vision, CNNs]
title: MIT 6.S191 - Lecture 3 - Convolutional Neural Networks
summary: 'Personal notes of `MIT 6.S191`, Lecture 3: Convolutional Neural Networks'
reading_time: yes
image:
  caption: ''
  focal_point: ''
  preview_only: true
type: today-i-learned
draft: false
---

# <span style="color:#FF9F1D">  Computer Vision Introduction </span>

We can train computers to understand the world of images, mapping where things are, what actions are taking place, and making them to predict and anticipate events in the world. For example, in this image, the computer can pick up that people are crossing the street, so the black car must be not moving.

{{< figure src="images/L3_cars.png" title="Objects picked up by computer vision" lightbox="true" >}}

# <span style="color:#FF9F1D">  What computers *see* </span>

Task that for us are trivial, for a computer is not. To a computer, the images are 2-dimensional arrays of numbers.

Taking the following image, we are able to see that is a Lincoln portrait but the computer sees a 1080x1080x3 vector of numbers.

{{< figure src="images/L3_lincoln.png" title="Lincoln image converted to numerical matrix taking the RGB color of the image pixels" lightbox="true" >}}

The classification of an image by a computer is made by picking up clues, or features, from the image. If the particular features of the image are more present in Lincoln images, it will be classified as Lincoln.

The algorithm, to perform this task well, should be able to differentiate between unique features and modifications of the same features. For example, it should classify as "Dog" a photo of dogs taken from different angles or a dog hidden in a tree.

{{< figure src="images/L3_transformation_images.png" title="Image transformations" lightbox="true" >}}

The computer must be invariant of all those variations, as humans recognize the same image changing its viewpoint or scale.

# <span style="color:#FF9F1D">  Learning Visual features </span>

**Computers learn hierarchically from the features** in an image. For example, in face recognition the algorithm learn in order:

1. Facial structure.
2. Eyes, ears, nose.
3. Edges, dark spots
4. ...

A fully connected neural network can take as input an image in the shape of a 2D number array, and classify it. What would be the problem of using a Multilayer Perceptron to classify images?

It's not able to capture is no **spatial information**.

If each feature of the image is an individual characteristic, all the connections between the image characteristics are lost. For example, a MLP architecture is not able to pick that the inner array of pixels the ears must be close to the outer array of pixels of the facial structure.

How can we use spatial structure in the input to inform the architecture of the network?

# <span style="color:#FF9F1D">  Patching </span>

Spatial 2D pixel arrays are correlated to each other. By using a spatial structure, it would preserve the correlation of the pixels and its spatial architecture.

We can think about a neural network architecture that takes different parts of the images in different layers and connects somehow the images. How would looks like?

In a neural network with spatial structure each neuron takes a small pixel of the entire image and try to extract it's feature information. Only a small region of the image, a **patch**, affects a concrete neuron and not the entire image.

{{< figure src="images/L3_patches.png" title="Spatial structure of CNN" lightbox="true" >}}

**The next neuron afterwards takes a shifted patch of pixels. The process is repeated for all the neurons until the entire image is taken as input by patches**.

As you can see in the image below , some of the patched pixels took from the first neuron in the left overlap some of the pixels pached in the right neuron.

{{< figure src="images/L3_patches_connected.png" title="Capturing image structure by patching" lightbox="true" >}}

The overlaping of pixels preserves the spatial component of the image. Every patch is intended to reveal features characteristic of the image.

But...how the algorithm learn the features? How it knows to detect the ears or eyes in a patch? The process is called *local feature extraction*.

# <span style="color:#FF9F1D">  Local feature Extraction </span>

The neural network identify the features patches by weigthing the pixels.

Take the following image. The idea is that the neural network have to classify the right image as an X or not a X.

{{< figure src="images/L3_xisx.png" title=" " lightbox="true" >}}

While for us humans is simple to see that is an X, the pixel arrays do not match. After all, computers cannot see images, only arrays of numbers that do not match.

By the process of patching, the neural network takes images with different pixel position that share same features:

{{< figure src="images/L3_xisxfeatures.png" title="" lightbox="true" >}}

Multiple patches in the X images are similar, or equal.

**How the model calculates this similarity?**

By **the convolutional operation**. While the name seems scary, it is just multiplying each pixel value elementwise between the filter matrix (*real X patch*) and the patch of the input image, and adding the outputs together.

{{< figure src="images/L3_convolutional_operation.png" title="The Convolutional Operation" lightbox="true" >}}

In other words, comparing the pixels between the *"proper X patch"* and the input patch that "*might or might not be an X patch*", in an anumerical way.

By going through local patches, the algorithm can identify and extract local features for each patch:

![](./images/L3_convolutional_operation_gif.gif "The Convolutional Operation for a whole image")

The end matrix from the convolutional operation is called **feature map**, as it mapped the features of the input image.

# <span style="color:#FF9F1D">  Convolutional Neural Netowrk operations </span>

CNNs are neural networks that apply the concept of patching, and are able to learn from spatial numerical arrays. **The word *Convolutional* is a way too say that this neural network architecture handles cross-correlated 2D arrays of numbers.**

Three CNN core operations are:

1. Convolution.
2. Apply a non-linear filter, often ReLU.
3. Pooling: a downsampling operation that allows to scale down the size of each feature map.

{{< figure src="images/L3_CNN.png" title="CNN structure" lightbox="true" >}}

**1. Convolution, or Convolutional Operations.**

The operation described in the above section. Each neuron takes **only the input from the patch**, computes the weighted sum, and applies bias that passes through a non-linear function (as usual in NN). Every neuron takes a different shifted patch.

![](./images/L3_feature_map.gif "How feature maps are created")

Take into account that there are not only one feature map in the neural network. **A feature map is specific for a feature**. As images have multiple features, multiple feature map or layers are needed.

Think about a human portrait. Taking only the feature *"oval shape of the face"* the algorithm could confuse a potato as a human face, as is oval as well.

By applying multiple filters, or layers, the CNN learns hierarchically from the features in an image.

**2. ReLU filter.**

After each convolutional operation, it needed to apply a ReLU activation function to the output volume of that layer.

**Why using a ReLU activation function?**

For any given neuron in the hidden layer, there are two possible (fuzzy) cases: either that neuron is relevant, or it isn’t. We need a function that shuts down the non-relevant neurons that do not contain a positive value.

ReLU replaces all the negative values with zero and keeps all the positive values with whatever the value was.

Think it this way: if the output of the convolutional operation is negative it means that the sample image patch doesn't look similar to the real image patch. We don't care how different it looks (how negative is the output), we only want that this neuron is not taken into account to train the model.

ReLU is also computationally cheap in comparison with other non-linear functions. It involves only a comparison between its input and the value 0.

**3. Pooling.**

Pooling is an operation to **reduce the dimensionality** of the inputs while still **preserving spatial invariants**. For example, a MaxPool2D takes a 4x4 patch matrix and convert it into a 2x2 patch by taking only the maximum value of each patch:

{{< figure src="images/L3_maxpool.png" title="MaxPool2D visual representation" lightbox="true" >}}


# <span style="color:#FF9F1D">  Convolutional Neural Netowrka for Image Classification </span>

Using CNNs for image classification can be broken down into 2 parts: learning and classification.

**1. Feature learning.**

The convolutional, ReLU and pooling matrix operations, the model to learn the features from an images. These feature maps get the important features of an image in the shape of weighted 2D arrays.

For example, a CNN architecture can learn from a set of images of cars and then distinguish between *car* features and *not car* features using the three key operations, but is still unable to classify images into labels.

**2. Classification part.**

**The second part of the CNN structure is using a second normal MPL to classify the label of the image**. After capturing the features of a car by convolutional operations and pooling, the lower-dimensional feature arrays feed this neural network to perform the classification.

{{< figure src="images/L3_CNN_classification_prob.png" title="CNNs for Classification: Class Probabilities" lightbox="true" >}}


**Why not using a second CNN structure or any other NN complex architecture?**

Because you don't need a neural network that handle sense of space or cross-corrlation for this task. It is a simple classification task. The inputs are not even an image anymore, they are features coded as number vectors. They don't need patching.

**Softmax function**

Given that the classification is into more than one category, the neural network output is filtered with a **softmax non-linear function to get the results in terms of probabilities**. The output of a softmax represents a categorical probability distribution. Following the car classification example, if the input image is a car it could give a 0.85 probability of being a car,  0.05 of being a van, a 0.01 of being a truck, and so forth.


# <span style="color:#FF9F1D">  Code example </span>

{{% alert look%}}
CNN "vehicle classifier" in Tensorflow:

  ***filters*** refers to the number of feature maps. For the first layer we set 32 feature maps, for the second 64.

  ***kernel_size*** refers to the height and width of the 2D convolution window. 3 means 3x3 pixel window patching.

  ***strides*** refers to how far the pooling window moves for each pooling step. With stride 2, the neurons moves in 2x2 pixels windows.

  ***pool_size*** refers to the window size over which to take the maximum when calculating the pooling operation. With 2, it will take the max value over a 2x2 pooling window.

  ***units*** refers to the number of outputs. 10 lasting outputs representing the 10 classes of vehicles.


{{% /alert %}}

```python
import tensorflow as tf

def vehicles_classifier_CNN():
  model = tf.keras.Sequential([

  ########First part: Feature learning ########

  ## CONVOLUTION + RELU
  tf.keras.layer.Conv2D(filters = 32,
                        kernel_size = 3,
                        activation = 'relu'),
  ## POOLING
  tf.keras.layer.MaxPool2D(pool_size = 2, strides = 2),
  ## CONVOLUTION + RELU
  tf.keras.layer.Conv2D(filters = 64,
                        kernel_size = 3,
                        activation = 'relu'),
  ## POOLING
  tf.keras.layer.MaxPool2D(pool_size = 2, strides = 2),

  ######## Second part: Classification ########

  ## FLATTEN
  tf.keras.layer.Flatten(),
  ## FULLY CONNECTED
  tf.keras.layer.Dense(units = 1024, activation = 'relu'),
  ## SOFTMAX
  tf.keras.layer.Dense(units = 10, activation = 'softmax')
  ])

  return model

```
