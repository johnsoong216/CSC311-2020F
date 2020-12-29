'''
Question 2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

from q1 import logsumexp_stable
import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    one_hot_labels = np.eye(len(np.unique(train_labels)))[train_labels.astype(int)]
    means = (one_hot_labels.T @ train_data)/np.sum(one_hot_labels, axis=0, keepdims=True).T
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((len(np.unique(train_labels)), train_data.shape[1], train_data.shape[1]))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    identity_coef = 0.01

    for i in range(len(np.unique(train_labels))):
        covariances[i] = ((train_data[train_labels == i] - means[i]).T @ (train_data[train_labels == i] - means[i])) \
                         / np.sum(train_labels == i) + np.eye(train_data.shape[1]) * identity_coef
    return covariances


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    exp_term = np.zeros(shape=(len(digits), means.shape[0]))
    const_term = -means.shape[1]/2 * np.log(2 * np.pi) - 1/2 * np.log(np.linalg.det(covariances))

    for j in range(means.shape[0]):
        exp_term[:, j] += -1/2 * np.diag((digits - means[j]) @ np.linalg.inv(covariances[j]) @ (digits - means[j]).T)
    return const_term + exp_term


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    prob_y_given = 0.1
    prob_y_vec = np.repeat(prob_y_given, means.shape[0])
    gen_likelihood = generative_likelihood(digits, means, covariances)
    return gen_likelihood + np.log(prob_y_vec) - logsumexp_stable(prob_y_vec + gen_likelihood, axis=1).reshape(-1, 1)


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute as described above and return
    one_hot_labels = np.eye(len(np.unique(labels)))[labels.astype(int)]
    return np.mean(np.sum(cond_likelihood * one_hot_labels, axis=1))

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)


def calc_accuracy(y_pred, y_target):
    return np.mean(y_pred == y_target)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('hw4digits')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    avg_likelihood_train = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    avg_likelihood_test = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    train_pred = classify_data(train_data, means, covariances)
    test_pred = classify_data(test_data, means, covariances)
    print("Train Avg Log Likelihood: ", avg_likelihood_train)
    print("Test Avg Log Likelihood: ", avg_likelihood_test)
    print("Train Accuracy: ", calc_accuracy(train_pred, train_labels))
    print("Test Accuracy: ", calc_accuracy(test_pred, test_labels))

    # Plot Eigenvalues
    eig_val, eig_vec = np.linalg.eig(covariances)

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(18, 9))
    for i in range(10):
        ax[i//5][i%5].imshow(np.reshape(eig_vec[i, :, eig_val[i].argmax()], (8, 8)), cmap='gray');
        ax[i//5][i%5].set_title(f"Digit {i}")
    fig.savefig("2c.png")


if __name__ == '__main__':
    main()
