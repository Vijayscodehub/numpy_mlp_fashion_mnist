# NumPy MLP From Scratch: Fashion-MNIST

## Goal

Implement the learning part of a neural network without PyTorch autograd or
PyTorch optimizers. Train a small multilayer perceptron (MLP) using NumPy on
Fashion-MNIST, then explain how each weight and bias changes during training.

## Project Scope

Use Fashion-MNIST's 60,000 training images and 10,000 official test images.
Create a reproducible split:

```text
60,000 training images
    -> 55,000 training images
    ->  5,000 validation images

10,000 official test images
    -> untouched until final evaluation
```

Build one network only:

```text
784 input pixels
    -> linear layer: W1, b1
    -> ReLU
    -> linear layer: W2, b2
    -> softmax probabilities for 10 classes
```

## Required Plan

1. Load Fashion-MNIST, flatten each image to 784 values, and normalize pixel
   values to the range `[0, 1]`.
2. Initialize `W1`, `b1`, `W2`, and `b2` manually with NumPy.
3. Implement the forward pass: linear layers, ReLU, softmax, and cross-entropy
   loss.
4. Implement backpropagation to calculate `dW1`, `db1`, `dW2`, and `db2`.
5. Implement mini-batch SGD to update each parameter with a learning rate.
6. After the first working model, use a numerical gradient check on a few
   parameter values to verify the manual gradients.
7. Use validation accuracy and loss to choose the learning rate, hidden-layer
   size, and number of epochs. Change one setting at a time.
8. Evaluate the selected model once on the untouched test set. Save learning
   curves, final accuracy, and a confusion matrix.

## Current Progress

Completed so far:

- Prepared a fixed 55,000 / 5,000 / 10,000 train, validation, and test split.
- Built the MLP forward pass, cross-entropy loss, manual backpropagation, and
  mini-batch SGD with NumPy.
- Verified one manual gradient against a numerical gradient estimate.
- Trained for 10 epochs and evaluated validation data after each epoch.

Current baseline settings:

```text
Hidden neurons: 128
Batch size: 128
Learning rate: 0.01
Epochs: 10
```

Best validation result:

```text
Loss:     0.4832
Accuracy: 83.26%
Epoch:    10
```

The next experiment changes only the learning rate to `0.001`. Validation
results will decide which setting is used for the one final test evaluation.

## Out Of Scope

- PyTorch `nn.Module`, autograd, or optimizers for the MLP implementation.
- CNNs, dropout, batch normalization, data augmentation, or hyperparameter
  sweeps.
- TensorBoard, unless it is added after the core implementation works.

## Definition Of Done

I can point to the code that initializes every weight and bias, calculates its
gradient through backpropagation, and updates it with gradient descent. I can
show that numerical and analytical gradients agree closely, and I can report a
single final test result from an untouched test set.
