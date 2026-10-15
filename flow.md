# Code Flow

This is the planned route data takes through the project. The names are a guide
for the functions we will create; they do not exist yet.

## Big Picture

```text
python -m src.train
        |
        v
main()
        |
        +--> load_and_split_data()
        |        |
        |        +--> train images and labels
        |        +--> validation images and labels
        |        +--> untouched test images and labels
        |
        +--> initialize_parameters()
        |        |
        |        +--> W1, b1, W2, b2
        |
        +--> train_model()
        |        |
        |        +--> forward_pass()
        |        +--> cross_entropy_loss()
        |        +--> backward_pass()
        |        +--> sgd_step()
        |        +--> validation_metrics()
        |
        +--> evaluate_test_set()
                 |
                 +--> final accuracy and confusion matrix
```

## Step By Step

### 1. `main()` starts the project

Think of `main()` as the teacher opening the lesson. It prepares the data,
creates the model's blank memory, runs practice, then gives the final exam.

**Entry:** `python -m src.train`  
**Exit:** saved final metrics and figures

### 2. `load_and_split_data()` prepares the examples

This function gets images and labels, turns each `28 x 28` image into a row of
784 numbers, and scales pixel values to `[0, 1]`.

```text
picture of a shoe -> 784 numbers -> matching label: "shoe"
```

It returns three separate groups: training for practice, validation for choices,
and test for the final exam.

**Input:** dataset files and a random seed  
**Output:** `train`, `validation`, and `test` image/label pairs

### 3. `initialize_parameters()` creates the adjustable knobs

`W1`, `b1`, `W2`, and `b2` are the knobs the model can learn to turn. They start
as small random numbers or zeros because the model does not know Fashion-MNIST
yet.

**Input:** input size, hidden size, class count, and seed  
**Output:** a parameter dictionary containing `W1`, `b1`, `W2`, and `b2`

### 4. `forward_pass()` makes a guess

A batch of image numbers travels through the two layers:

```text
image numbers -> first layer -> ReLU -> second layer -> class probabilities
```

Like a child sorting pictures into ten boxes, the model puts a probability in
each box. The largest probability is its guess.

**Input:** images and parameters  
**Output:** probabilities plus saved intermediate values needed by backpropagation

### 5. `cross_entropy_loss()` says how wrong the guess was

This function compares the probability assigned to the correct class with the
true label. Confident wrong guesses receive a larger penalty.

**Input:** predicted probabilities and true labels  
**Output:** one average loss number

### 6. `backward_pass()` finds which knobs caused the mistake

This is the learning detective. Starting from the loss, it walks backward and
asks how changing each weight or bias would change the mistake.

```text
loss -> dW2 and db2 -> hidden layer -> dW1 and db1
```

**Input:** forward-pass saved values, labels, and parameters  
**Output:** gradients `dW1`, `db1`, `dW2`, and `db2`

### 7. `sgd_step()` makes a tiny correction

Each parameter moves a small amount in the direction that should reduce loss:

```text
new weight = old weight - learning_rate * gradient
```

**Input:** parameters, gradients, and learning rate  
**Output:** updated parameters

### 8. `train_model()` repeats practice

It sends one small batch through steps 4 to 7 many times. At the end of each
epoch, it calls the forward pass on validation data without changing weights.

**Input:** training data, validation data, parameters, and settings  
**Output:** trained parameters and training/validation history

### 9. `evaluate_test_set()` gives the final exam

Only after settings are selected, this function uses the final parameters to
make predictions for the untouched test set. It produces the final accuracy and
confusion matrix; it must not trigger another model change.

**Input:** selected parameters and untouched test data  
**Output:** final metrics and error-analysis figures

## Connections To Protect

- `forward_pass()` must use the same parameter shapes that
  `initialize_parameters()` creates.
- `backward_pass()` needs the exact intermediate values produced by the matching
  `forward_pass()`.
- `sgd_step()` changes parameters only during training, never validation or
  testing.
- Validation selects settings; testing only reports the final result.
