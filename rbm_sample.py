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

def visualize(M, N, x, fig_name):
	fig, ax = plt.subplots(figsize=(N,M))
	digits = np.vstack([np.hstack([np.reshape(x[:, i*N+j],(28,28)) 
	                               for j in range(N)]) for i in range(M)])
	ax.imshow(255-digits, cmap=plt.get_cmap('gray'))
	# plt.show()
	plt.savefig(fig_name)

train_images, train_labels = load_data("digitstrain.txt")
valid_images, valid_labels = load_data("digitsvalid.txt")

num_hidden_units = 100
epochs = 30000
learn_rate = 0.2
k = 20

variance1 = math.sqrt(6.0) / (784.0 + num_hidden_units)
W1 = weight_variables(num_hidden_units, 784, variance1)
b1 = bias_variables(num_hidden_units, 1, 0)
c1 = bias_variables(784, 1, 0)

num_examples = 50
images_ones = np.ones((num_examples, 1))

for epoch in range(0, epochs):
    images, _ = get_batch(num_examples)
    images = sample(images, 784)
    h, h_tilde, x_tilde, x_bin = CD_k(images, k)

    W1_grad = (h.dot(images.T) - h_tilde.dot(x_bin.T)) / num_examples
    b1_grad = (h - h_tilde).dot(images_ones) / num_examples
    c1_grad = (images - x_bin).dot(images_ones) / num_examples

    W1 += learn_rate * W1_grad
    b1 += learn_rate * b1_grad
    c1 += learn_rate * c1_grad

images, _ = get_batch(100)
# randomize
images = sample(images + np.random.rand(images.shape[0], images.shape[1]) - 0.5, 784)
_, _, images_tilde, images_bin = CD_k(images, 1000)

visualize(10, 10, images_tilde, "(c) visualization of sampled digits.png")
visualize(10, 10, images_bin, "(c) visualization of sampled digits bin.png")
