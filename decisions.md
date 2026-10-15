# Decision Log

Record decisions that affect how results should be interpreted. Update this file
before changing a core choice, rather than relying on memory.

| ID | Decision | Reason | Consequence |
| --- | --- | --- | --- |
| D1 | Use Fashion-MNIST. | It is small, visual, and simple enough to debug manual backpropagation. | Results teach neural-network mechanics, not high image accuracy. |
| D2 | Use NumPy for the MLP math. | We need to see weights, biases, gradients, and updates directly. | No `nn.Module`, autograd, or optimizer may implement the model. |
| D3 | Use a single hidden layer with ReLU. | It is the smallest network that demonstrates nonlinear learning. | The model has only `W1`, `b1`, `W2`, and `b2`. |
| D4 | Use a fixed 55k/5k/10k split. | Training learns parameters, validation selects settings, and test estimates final generalization. | The official test set stays untouched until the final run. |
| D5 | Scale pixels to `[0, 1]`. | Inputs on a predictable scale make gradient descent easier to train and reason about. | Apply exactly the same transformation to every split. |
| D6 | Use softmax and cross-entropy. | This is the standard match for one-label, ten-class classification. | The output layer returns ten scores per image. |
| D7 | Start with mini-batch SGD. | It makes the update rule transparent before using more complex optimizers. | Learning rate is the first tuning choice. |
| D8 | Run a numerical gradient check. | A decreasing loss alone cannot prove manual backpropagation is correct. | Compare a few analytical gradients with finite-difference estimates. |
| D9 | Select settings using validation data only. | Test-informed tuning gives an overly optimistic final result. | Run final test evaluation once after selection. |

## New Decision Template

Add a row when changing a core choice such as the split, model shape, loss,
optimizer, or evaluation protocol.

| ID | Decision | Reason | Consequence |
| --- | --- | --- | --- |
| D10 | _What changes?_ | _Why change it?_ | _How does it affect fair comparison or interpretation?_ |
