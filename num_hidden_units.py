import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return np.transpose(labels_one_hot)

def load_data(path):
    data = pd.read_csv(path).fillna('')
    raw = data.as_matrix()
    images = np.transpose(raw[:, 0:784])
    labels = dense_to_one_hot(np.transpose(np.matrix(raw[:, 784]).astype('int')))
    return (images, labels)

def weight_variables(dim1, dim2, variance):
    matrix = np.multiply(2, np.random.rand(dim1, dim2))
    matrix = np.multiply(np.subtract(matrix, 1), variance)
    return matrix

def bias_variables(dim1, dim2, variance):
    matrix = np.zeros((dim1, dim2))
    matrix = np.multiply(matrix, variance)
    return matrix

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x)) # sigmoid

def cross_entropy(y, truth):
    return -np.add(np.multiply(truth, np.log(y)), \
    	np.multiply(-np.subtract(truth, 1), np.log(-np.subtract(y, 1))))

def sample(x, dim):
    return (x > np.random.rand(dim, 1)).astype('int')
    # return (x > 0.5).astype('int') # 0.5 as threshold

def get_batch(batch_size=50):
    indexes = np.random.randint(0, 2999, batch_size)
    return train_images[:, indexes], train_labels[:, indexes]

def CD_k(x, k):
    h = sigmoid(W1.dot(x) + b1)
    h_tilde = h
    for step in range(0, k):
        h_bin = sample(h_tilde, num_hidden_units)
        x_tilde = sigmoid(W1.T.dot(h_bin) + c1)
        x_bin = sample(x_tilde, 784)
        h_tilde = sigmoid(W1.dot(x_bin) + b1)
    return h, h_tilde, x_tilde, x_bin

def plot_entropy(train_entropy, valid_entropy, train_label, valid_label, fig_name):
    epoch_range = np.arange(len(valid_entropy))
    plt.xlabel('epoch time')
    plt.ylabel('entropy')
    plt.plot(epoch_range, train_entropy, label=train_label)
    plt.plot(epoch_range, valid_entropy, label=valid_label)
    plt.legend()
    # plt.show()
    plt.savefig(fig_name)
    plt.gcf().clear()

def visualize(M, N, x, fig_name):
	fig, ax = plt.subplots(figsize=(N,M))
	digits = np.vstack([np.hstack([np.reshape(x[:, i*N+j],(28,28)) 
	                               for j in range(N)]) for i in range(M)])
	ax.imshow(255-digits, cmap=plt.get_cmap('gray'))
	# plt.show()
	plt.savefig(fig_name)

def feed_forward_auto(x, W1, b1, W2, b2):
    h1 = sigmoid(np.add(np.matmul(W1, x), b1))
    y = np.add(np.matmul(W2, h1), b2)
    return h1, y

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, 0)
    return np.divide(exp_x, sum_exp_x)

def error_rate(y, truth):
    return np.mean(np.argmax(truth, 0) != np.argmax(y, 0))

def evaluate_auto(y, truth):
    # loss = cross_entropy(y, truth)
    loss = (y - truth) **2 / 2
    return loss

def gen_mask(dim1, dim2, dropout):
    mask = np.random.rand(dim1, dim2)
    mask[mask < dropout] = 0
    mask[mask >= dropout] = 1
    return mask

np.random.seed(10807)

train_images, train_labels = load_data("digitstrain.txt")
valid_images, valid_labels = load_data("digitsvalid.txt")
test_images, test_labels = load_data("digitstest.txt")


epochs = 3000
learn_rate = 0.2
dropout = 0.0
momentum = 0.0
weight_decay = 0.0
k = 1

for num_hidden_units in [50, 100, 200, 500]:

    # rbm 
    print "RBM"

    variance1 = math.sqrt(6.0) / (784.0 + num_hidden_units)
    W1 = weight_variables(num_hidden_units, 784, variance1)
    b1 = bias_variables(num_hidden_units, 1, 0)
    c1 = bias_variables(784, 1, 0)

    num_examples = 50
    images_ones = np.ones((num_examples, 1))
    interval = 300
    train_entropy = [0] * (epochs / interval)
    valid_entropy = [0] * (epochs / interval) 
    record = 0

    for epoch in range(0, epochs):
        images, _ = get_batch(num_examples)
        images = sample(images, 784)
        h, h_tilde, x_tilde, x_bin = CD_k(images, k)
        if epoch % interval == 0:
            _, _, x_train, _ = CD_k(train_images, k)
            loss = np.sum(cross_entropy(x_train, train_images)) / train_images.shape[1]
            train_entropy[record] = loss
            print "epoch", epoch, "training loss", loss
            _, _, x_valid, _ = CD_k(valid_images, k)
            loss = np.sum(cross_entropy(x_valid, valid_images)) / valid_images.shape[1]
            valid_entropy[record] = loss
            print "epoch", epoch, "validation loss", loss
            record += 1

        W1_grad = (h.dot(images.T) - h_tilde.dot(x_bin.T)) / num_examples
        b1_grad = (h - h_tilde).dot(images_ones) / num_examples
        c1_grad = (images - x_bin).dot(images_ones) / num_examples

        W1 += learn_rate * W1_grad
        b1 += learn_rate * b1_grad
        c1 += learn_rate * c1_grad

    plot_entropy(train_entropy, valid_entropy, "train", "valid", "(g) rbm entropy_{0}.png".format(num_hidden_units))

    # auto-encoder

    print "Auto-encoder"

    variance1 = math.sqrt(6.0) / (784.0 + num_hidden_units)
    W1 = weight_variables(num_hidden_units, 784, variance1)
    b1 = bias_variables(num_hidden_units, 1, 0)
    W2 = W1.T
    b2 = bias_variables(784, 1, 0)

    num_examples = 50
    images_ones = np.ones((num_examples, 1))

    last_W2 = np.zeros(W2.shape)
    last_b2 = np.zeros(b2.shape)
    last_W1 = np.zeros(W1.shape)
    last_b1 = np.zeros(b1.shape)
    record = 0

    for epoch in range(0, epochs):
        images, _ = get_batch(num_examples)
        labels = images

        h1, y = feed_forward_auto(images, W1, b1, W2, b2)
        images_transposed = np.transpose(images)

        if epoch % interval == 0:
            _, y_train = feed_forward_auto(train_images, W1, b1, W2, b2)
            train_loss = np.sum(evaluate_auto(y_train, train_images)) / train_images.shape[1]
            print "epoch", epoch, "training loss", train_loss
            _, y_valid = feed_forward_auto(valid_images, W1, b1, W2, b2)
            valid_loss = np.sum(evaluate_auto(y_valid, valid_images)) / valid_images.shape[1]
            print "epoch", epoch, "valid loss", valid_loss
            train_entropy[record] = train_loss
            valid_entropy[record] = valid_loss
            record += 1

        h1_tranposed = np.transpose(h1)
        
        loss_derivative = np.subtract(y, labels) / num_examples

        output = loss_derivative # np.multiply(loss_derivative, np.multiply(y, (1-y)))
        hidden = np.multiply(np.matmul(np.transpose(W2), output), np.multiply(h1, (1-h1)))

        W2_gradient = np.matmul(output, h1_tranposed) + (momentum * last_W2) + weight_decay * W2
        b2_gradient = np.matmul(output, np.ones((h1_tranposed.shape[0], 1))) + (momentum * last_b2)
        b2 = np.subtract(b2, np.multiply(learn_rate, b2_gradient))

        W1_gradient = np.matmul(hidden, images_transposed) + (momentum * last_W1) + weight_decay * W1
        b1_gradient = np.matmul(hidden, images_ones) + (momentum * last_b1)
        b1 = np.subtract(b1, np.multiply(learn_rate, b1_gradient))
        
        W1 = np.subtract(W1, np.multiply(learn_rate/2.0, (W1_gradient + W2_gradient.T)))
        W2 = W1.T

        last_W2 = W2_gradient
        last_b2 = b2_gradient
        last_W1 = W1_gradient
        last_b1 = b1_gradient

    plot_entropy(train_entropy, valid_entropy, "train", "valid", "(g) AE entropy_{0}.png".format(num_hidden_units))

    # denoising 

    print "Denoising auto-encoder"

    variance1 = math.sqrt(6.0) / (784.0 + num_hidden_units)
    W1 = weight_variables(num_hidden_units, 784, variance1)
    b1 = bias_variables(num_hidden_units, 1, 0)
    W2 = W1.T
    b2 = bias_variables(784, 1, 0)

    num_examples = 50
    images_ones = np.ones((num_examples, 1))

    last_W2 = np.zeros(W2.shape)
    last_b2 = np.zeros(b2.shape)
    last_W1 = np.zeros(W1.shape)
    last_b1 = np.zeros(b1.shape)
    record = 0

    for epoch in range(0, epochs):
        images, _ = get_batch(num_examples)
        labels = images
        images = np.multiply(images, gen_mask(images.shape[0], images.shape[1], 0.5))

        h1, y = feed_forward_auto(images, W1, b1, W2, b2)
        images_transposed = np.transpose(images)

        if epoch % interval == 0:
            _, y_train = feed_forward_auto(train_images, W1, b1, W2, b2)
            train_loss = np.sum(evaluate_auto(y_train, train_images)) / train_images.shape[1]
            print "epoch", epoch, "training loss", train_loss
            _, y_valid = feed_forward_auto(valid_images, W1, b1, W2, b2)
            valid_loss = np.sum(evaluate_auto(y_valid, valid_images)) / valid_images.shape[1]
            print "epoch", epoch, "valid loss", valid_loss
            train_entropy[record] = train_loss
            valid_entropy[record] = valid_loss
            record += 1

        h1_tranposed = np.transpose(h1)
        
        loss_derivative = np.subtract(y, labels) / num_examples

        output = loss_derivative # np.multiply(loss_derivative, np.multiply(y, (1-y)))
        hidden = np.multiply(np.matmul(np.transpose(W2), output), np.multiply(h1, (1-h1)))

        W2_gradient = np.matmul(output, h1_tranposed) + (momentum * last_W2) + weight_decay * W2
        b2_gradient = np.matmul(output, np.ones((h1_tranposed.shape[0], 1))) + (momentum * last_b2)
        b2 = np.subtract(b2, np.multiply(learn_rate, b2_gradient))

        W1_gradient = np.matmul(hidden, images_transposed) + (momentum * last_W1) + weight_decay * W1
        b1_gradient = np.matmul(hidden, images_ones) + (momentum * last_b1)
        b1 = np.subtract(b1, np.multiply(learn_rate, b1_gradient))
        
        W1 = np.subtract(W1, np.multiply(learn_rate/2.0, (W1_gradient + W2_gradient.T)))
        W2 = W1.T

        last_W2 = W2_gradient
        last_b2 = b2_gradient
        last_W1 = W1_gradient
        last_b1 = b1_gradient

    plot_entropy(train_entropy, valid_entropy, "train", "valid", "(g) DAE entropy_{0}.png".format(num_hidden_units))
