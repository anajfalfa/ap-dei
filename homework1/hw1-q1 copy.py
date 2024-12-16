#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """

        scores = np.dot(self.W, x_i.T)  # (n_classes x n_examples)
        y_pred = scores.argmax(axis=0)  # (n_examples)
        if y_pred != y_i:
            self.W[y_i,:] = self.W[y_i,:] + x_i
            self.W[y_pred,:] = self.W[y_pred,:] - x_i
        #raise NotImplementedError # Q1.1 (a)


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        scores = np.expand_dims(self.W.dot(x_i), axis = 1)
        p_scores = np.exp(scores)/sum(np.exp(scores)) # softmax 
        e_y = np.zeros((np.size(self.W, 0),1))
        e_y[y_i] = 1
        grad = (p_scores - e_y).dot(np.expand_dims(x_i, axis = 1).T)
        if l2_penalty == 0:
            #scores = np.dot(self.W, x_i.T)
            #stochastic grad descent:
            self.W = self.W - learning_rate*grad
        else:
            #stochastic grad descent:
            self.W = (1- learning_rate   * l2_penalty) * self.W - learning_rate*grad
        #raise NotImplementedError # Q1.2 (a,b)

def relu_activation (z):
        return np.maximum(0,z)

def softmax(vector):
    # 0 ou 1
    #exp_z = np.exp(Z2 - np.max(Z2, axis=1, keepdims = True))
    #softmax = exp_z / (np.sum(exp_z, axis=1, keepdims = True)+1e-6)

    softmax = []
    for z in vector:
        exp_z = np.exp(z - np.max(z))  # Subtrair o máximo
        sm = exp_z / (sum(exp_z) )  # Divisão com epsilon + 1e-15
        #print(sm)
        softmax += [sm]
    return softmax

def cross_entropy(pred, true):
    #loss = np.log(y_pred[yi])
    #loss = -e_y.dot(np.log(y_pred + epsilon))
    #loss = -np.mean(np.sum(e_y * np.log (y_pred + 1e-9), axis=1))
    #np.log(np.dot(pred, true.T))
    return -np.sum(true*np.log(pred))

class MLP(object):
    # feed-forward neural network
    # input layer --> hidden layer --> output layer

    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        mu, sigma = 0.1, 0.1
        self.W1 = np.random.normal(mu,sigma, (hidden_size, n_features))
        self.b1 = np.zeros((hidden_size,1))

        self.W2 = np.random.normal(mu,sigma, (n_classes, hidden_size))
        self.b2 = np.zeros((n_classes,1))

        #self.b1 = np.zeros((hidden_size,))
        #self.b2 = np.zeros((n_classes,))
        # gradient backpropagation train

        #raise NotImplementedError # Q1.3 (a)
    
    def predict(self, X):        
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        print("X no predict", X.shape)

        #Z1 = np.dot(self.W1, X.T) + self.b1[:, np.newaxis] # hidden x samples
        Z1 = np.dot(self.W1, X.T) + self.b1 # hidden x samples

        print("Z1",Z1.shape)
        H = np.maximum(0,Z1)
        print("H",H.shape)

        #H = np.maximum(0,Z1)
        #Z2 = np.dot(self.W2, H) + self.b2[:, np.newaxis] # classes x samples
        Z2 = np.dot(self.W2, H) + self.b2 # classes x samples
        print("Z2",Z2.shape)

        O = softmax(Z2.T)
        #print("softmax", softmax.shape)
        #print(softmax)
        # Hidden layer pre-activation: z(x)=W(1)x + b(1)
        # Hidden layer activation: h(x)=g(z(x)) component-wise
        # Output layer activation: f(x) =o(h(x)Tw(2)+b(2)) -- change multiple output units (6) -> P(y=c|x) + softmax
        
        #predictions = np.argmax(O, axis=1)
        predictions = np.argmax(O, axis=1)
        print("predictions", predictions.shape)

        return predictions
        #raise NotImplementedError # Q1.3 (a)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        #print("y_hat",y_hat.shape)
        
        #print("y",y.shape)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """
        total_loss = 0
        print("x", X.shape)
        for xi, yi in zip(X,y):
            e_y = np.zeros((np.size(self.W2, 0),1))
            e_y[yi] = 1
            #print("xi", xi.shape)

            # forward-pass
            #self.W1 * X + self.b1
            #print("W1", self.W1.shape)
            #print("b1",self.b1.shape)
            aaaa = np.dot(self.W1, np.expand_dims(xi, axis = 1))
            z_1 = aaaa + self.b1
            #print(aaaa.shape)
            ##print("z1", z_1.shape)

            #z_1 = np.dot(self.W1, np.expand_dims(xi, axis=1))+self.b1[:, np.newaxis]
            #np.expand_dims(xi, axis = 1)) + self.b1

            #z_1 = xi @ self.W1.T + self.b1
            h_1 = np.maximum (0,z_1)

            #self.W2 * h_1 + self.b2
            #z_2 = np.dot(self.W2, h_1) + self.b2[:, np.newaxis]
            z_2 = np.dot(self.W2, h_1) + self.b2
            #print("z2", z_2.shape)

            # 1*6

            #z_2 =  h_1 @ self.W2.T + self.b2
            '''exp_z = np.exp(z_2 - np.max(z_2, axis = 1, keepdims = True))
            softmax = exp_z / np.sum(exp_z, axis=1, keepdims = True)
            '''
            #pred_probs = softmax([z2])[0]   
            #print("softmax", softmax.shape)
            y_pred = softmax ([z_2])[0]
            #print("y_pred 6l", y_pred.shape)

            #print("softmax ok",np.sum(y_pred))
            epsilon=1e-6

            #compute the loss
            # 1x6 6x1
            #print(e_y, y_pred)
            y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # Stability trick
            loss = cross_entropy (y_pred, e_y) # ou yi
            total_loss += loss
            #print("loss", total_loss)
            #backpropagation
            grad_o = y_pred - e_y #gradient of the loss function at output (output error)

            #w2_grad = grad_o[:, None].dot(h[:, None].T)
            w2_grad = np.dot(grad_o, h_1.T)
            b2_grad = grad_o

            grad_h_a = np.dot(self.W2.T, grad_o) #hidden layer errors
            relu_derivative = (z_1 > 0).astype(float)  # Derivada da ReLU
            grad_h = grad_h_a * relu_derivative

            #w1_grad = grad_h[:, None].dot(xi[:, None].T)
            w1_grad = np.dot(grad_h, np.expand_dims(xi, axis = 1).T)
            b1_grad = grad_h

            self.W1 -= learning_rate * w1_grad
            self.b1 -= learning_rate * b1_grad
            self.W2 -= learning_rate * w2_grad
            self.b2 -= learning_rate * b2_grad

        #total_loss = total_loss / X.shape(0)

        return total_loss
    
        #y_true = true label (one hot encoded)
        #y_pred = predicted probability for class k
        #raise NotImplementedError # Q1.3 (a)


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp" # if model is different from mlp
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate =opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
