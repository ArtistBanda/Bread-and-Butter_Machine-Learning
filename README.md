# Bread-and-Butter_Machine-Learning

## About

Implementation of machine learning algorithms from scratch.

## Table of Contents

- [Bread-and-Butter_Machine-Learning](#bread-and-butter_machine-learning)
  - [About](#about)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Supervised Learning](#supervised-learning)
    - [Linear Model](#linear-model)
    - [Logistic Regression](#logistic-regression)
    - [Naive Bayes](#naive-bayes)
  - [Deep Learning](#deep-learning)
    - [Neural Network](#neural-network)
    - [Layers](#layers)
    - [Activation Functions](#activation-functions)
    - [Backward Activation Fucntions](#backward-activation-fucntions)
  - [Utils](#utils)
    - [Loss Functions](#loss-functions)
    - [Metrics](#metrics)
    - [Pre-Processing](#pre-processing)

## Installation

  ``` bash
  git clone https://github.com/ArtistBanda/Bread-and-Butter_Machine-Learning.git

  cd Bread-and-Butter_Machine-Learning

  python setup.py install
  ```

## Supervised Learning

### Linear Model

[code](bnbML/Supervised_Learning/LinearModel.py)  

[Linear Regression is a machine learning](https://www.scaler.com/topics/linear-regression-in-machine-learning/) algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables, they are considering and the number of independent variables being used.
[Further Read](https://www.geeksforgeeks.org/ml-linear-regression)

### Logistic Regression

[code](bnbML/Supervised_Learning/LogisticRegression.py)

It is just like [Linear Regression](#linear-model) except that the out is a discrete value instead of a continuous output and output is masked by an activation function like sigmoid or softmax.

### Naive Bayes

[code](bnbML/Supervised_Learning/NaiveBayes.py)

Naive Bayes classifiers are a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other. [Further Read](https://www.geeksforgeeks.org/naive-bayes-classifiers)

## Deep Learning

### Neural Network

[code](bnbML/Deep_Learning/NeuralNetwork.py)

Neural networks are a class of machine learning algorithms used to model complex patterns in datasets using multiple hidden layers and non-linear activation functions. A neural network takes an input, passes it through multiple layers of hidden neurons (mini-functions with unique coefficients that must be learned), and outputs a prediction representing the combined input of all the neurons. [Further Read](https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html#neural-network)

### Layers

[code](bnbML/Deep_Learning/Layers.py)

**Base Layer**

Bolier plate class which is parent for all layers below. Base class Layer from which all the layers will be derived from

**Input Layer**

Holds the data your model will train on. Each neuron in the input layer represents a unique attribute in your dataset (e.g. height, hair color, etc.).

**Dense Layer**

A Dense is one of the most simple layer which is used in Neural Networks.
It requires the number of neurons (units) it will have along the activation 
function it will use.

**Flatten Layer**

A layer class which reduces the previous layers output (usually a conv layer) into 
a straight linear array of size (n, 1)

**Convolutional Layer**

In CNN, a convolution is a linear operation that involves multiplication of weight (kernel/filter) with the input and it does most of the heavy lifting job.

Convolution layer consists of 2 major component 1. Kernel(Filter) 2. Stride

Kernel (Filter): A convolution layer can have more than one filter. The size of the filter should be smaller than the size of input dimension. It is intentional as it allows filter to be applied multiple times at difference point (position) on the input.Filters are helpful in understanding and identifying important features from given input. By applying different filters (more than one filter) on the same input helps in extracting different features from given input. Output from multiplying filter with the input gives Two dimensional array. As such, the output array from this operation is called “Feature Map”.
Stride: This property controls the movement of filter over input. when the value is set to 1, then filter moves 1 column at a time over input. When the value is set to 2 then the filer jump 2 columns at a time as filter moves over the input.



**Hidden Layers**

Sits between the input and output layers and applies an activation function before passing on the results. There are often multiple hidden layers in a network. In traditional networks, hidden layers are typically fully-connected layers — each neuron receives input from all the previous layer’s neurons and sends its output to every neuron in the next layer. This contrasts with how convolutional layers work where the neurons send their output to only some of the neurons in the next layer.

**Output Layers**

The final layer in a network. It receives input from the previous hidden layer, optionally applies an activation function, and returns an output representing your model’s prediction.



[Further Read](https://ml-cheatsheet.readthedocs.io/en/latest/layers.html)

### Activation Functions

[code](bnbML/Deep_Learning/ActivationFunctions.py)

Activation functions live inside neural network layers and modify the data they receive before passing it to the next layer. Activation functions give neural networks their power — allowing them to model complex non-linear relationships. By modifying inputs with non-linear functions neural networks can model highly complex relationships between features. Popular activation functions include relu and sigmoid. [Further Read](https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html#activation-functions)

### Backward Activation Fucntions

[code](bnbML/Deep_Learning/BackwardActivationFucntions.py)

These are derivatives of activation functions which are required to be calculated to perform backpropagation algorithm for training a Neural Network.
  
## Utils

### Loss Functions

[code](bnbML/Utils/LossFunctions.py)

Machines learn by means of a loss function. It’s a method of evaluating how well specific algorithm models the given data. If predictions deviates too much from actual results, loss function would cough up a very large number. Gradually, with the help of some optimization function, loss function learns to reduce the error in prediction
[Further Read](https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23)

### Metrics

[code](bnbML/Utils/Metrics.py)

To evaluate a model of how good it performs, different kinds of metrics like accuracy or F1 score are used.
[Further Read](https://towardsdatascience.com/metrics-to-evaluate-your-machine-learning-algorithm-f10ba6e38234)

### Pre-Processing

[code](bnbML/Utils/PreProcessing.py)

Functions which can be applied on the data, which increases the quality of it and can affect the ability of our model to learn.
