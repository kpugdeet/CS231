import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    # self.params['W1'] = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    # self.params['W2'] = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    pass
    tmp = X.dot(W1) + b1
    h_output = np.maximum(0.01 * tmp, tmp)  # Leaky RELU Activation
    # h_output = np.maximum (0, tmp) # RELU Activation
    # h_output = np.tanh(tmp) # Tanh Activation
    scores = h_output.dot(W2) + b2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    pass
    # SoftMax Score
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

    # Average cross-entropy loss
    corect_logprobs = -np.log(probs[range(N), y]) # Do log for only the collect class y = 1 dimension
    data_loss = np.sum(corect_logprobs) / N

    # L2 regularization
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)

    # Sum
    loss = data_loss + reg_loss
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    pass
    # http://cs231n.github.io/neural-networks-case-study/
    dscores = probs
    dscores[range(N), list(y)] -= 1
    dscores /= N
    grads['W2'] = h_output.T.dot(dscores) + reg * W2
    grads['b2'] = np.sum(dscores, axis=0)

    dh = dscores.dot(W2.T)
    # dh_ReLu = (h_output > 0) * dh
    dh_ReLu = (h_output >= 0) * dh + (h_output < 0) * dh * 0.01
    # dh_ReLu = (1.0 - np.tanh(h_output) ** 2) * dh
    grads['W1'] = X.T.dot(dh_ReLu) + reg * W1
    grads['b1'] = np.sum(dh_ReLu, axis=0)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, momentum=0.9,verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    self.params['VW2'] = 0
    self.params['VW1'] = 0
    self.params['cacheW2'] = 0
    self.params['cacheW1'] = 0
    self.params['MW2'] = 0
    self.params['MW1'] = 0

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      idx = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[idx]
      y_batch = y[idx]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      # Vanilla
      # self.params['W2'] += - learning_rate * grads['W2']
      # self.params['W1'] += - learning_rate * grads['W1']

      # Momentum
      # self.params['VW2'] = momentum * self.params['VW2'] - learning_rate * grads['W2']
      # self.params['W2'] += self.params['VW2']
      # self.params['VW1'] = momentum * self.params['VW1'] - learning_rate * grads['W1']
      # self.params['W1'] += self.params['VW1']

      # Nesterov
      # v_prevW2 = self.params['VW2']
      # self.params['VW2'] = momentum * self.params['VW2'] - learning_rate * grads['W2']
      # self.params['W2'] += -momentum * v_prevW2 + (1 + momentum) * self.params['VW2']
      # v_prevW1 = self.params['VW1']
      # self.params['VW1'] = momentum * self.params['VW1'] - learning_rate * grads['W1']
      # self.params['W1'] += -momentum * v_prevW1 + (1 + momentum) * self.params['VW1']

      # Adagrad
      # self.params['cacheW2'] += grads['W2'] ** 2
      # self.params['W2'] += - learning_rate * grads['W2'] / (np.sqrt(self.params['cacheW2']) + 1e-7)
      # self.params['cacheW1'] += grads['W1'] ** 2
      # self.params['W1'] += - learning_rate * grads['W1'] / (np.sqrt(self.params['cacheW1']) + 1e-7)

      # rmsProbs
      # self.params['cacheW2'] = 0.9 * self.params['cacheW2'] + (1 - 0.9) * grads['W2'] ** 2
      # self.params['W2'] += - learning_rate * grads['W2'] / (np.sqrt(self.params['cacheW2']) + 1e-7)
      # self.params['cacheW1'] = 0.9 * self.params['cacheW1'] + (1 - 0.9) * grads['W1'] ** 2
      # self.params['W1'] += - learning_rate * grads['W1'] / (np.sqrt(self.params['cacheW1']) + 1e-7)

      # Adam
      self.params['MW2'] = 0.9 * self.params['MW2'] + (1 - 0.9) * grads['W2']
      mtW2 = self.params['MW2'] / (1 - 0.9 ** (it+1))
      self.params['VW2'] = 0.999 * self.params['VW2'] + (1 - 0.999) * (grads['W2'] ** 2)
      vtW2 = self.params['VW2'] / (1- 0.999 ** (it+1))
      self.params['W2'] += - learning_rate * mtW2 / (np.sqrt(vtW2) + 1e-8)
      self.params['MW1'] = 0.9 * self.params['MW1'] + (1 - 0.9) * grads['W1']
      mtW1 = self.params['MW1'] / (1 - 0.9 ** (it+1))
      self.params['VW1'] = 0.999 * self.params['VW1'] + (1 - 0.999) * (grads['W1'] ** 2)
      vtW1 = self.params['VW1'] / (1 - 0.999 ** (it+1))
      self.params['W1'] += - learning_rate * mtW1 / (np.sqrt(vtW1) + 1e-8)

      self.params['b2'] += - learning_rate * grads['b2']
      self.params['b1'] += - learning_rate * grads['b1']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 10 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    h = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])
    scores = h.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


