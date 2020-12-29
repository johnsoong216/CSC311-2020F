import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

# Load Data
data_path = "hw1_data/"
data_train = {'X': np.genfromtxt(data_path + 'data_train_X.csv', delimiter=','),
              't': np.genfromtxt(data_path + 'data_train_y.csv', delimiter=',')}
data_test = {'X': np.genfromtxt(data_path + 'data_test_X.csv', delimiter=','),
              't': np.genfromtxt(data_path + 'data_test_y.csv', delimiter=',')}


def shuffle_data(data):
    """
    Shuffle data uniformly
    :param data: dict, input data of X and t
    :return: dict, with X,t
    """
    data_shf = {}
    idx = np.arange(0, data['X'].shape[0], 1)
    np.random.shuffle(idx)
    data_shf['X'] = data['X'][idx]
    data_shf['t'] = data['t'][idx]
    return data_shf


def split_data(data, num_folds, fold):
    """
    Split Data into Random Folds
    :param data: dict, X,t
    :param num_folds: int, number of folds
    :param fold: int, which fold to be used as CV
    :return: tuple of Dict, of CV dataset and Train dataset
    """
    data_rest = {}
    data_fold = {}

    data_len = data['X'].shape[0]
    data_x = np.split(data['X'], np.arange(int(data_len / num_folds), data_len, int(data_len / num_folds)))
    data_t = np.split(data['t'], np.arange(int(data_len / num_folds), data_len, int(data_len / num_folds)))

    data_fold['X'] = data_x[fold]
    data_fold['t'] = data_t[fold]

    data_rest['X'] = np.concatenate(data_x[:fold] + data_x[fold + 1:], axis=0)
    data_rest['t'] = np.concatenate(data_t[:fold] + data_t[fold + 1:], axis=0)
    return data_fold, data_rest


def train_model(data, lambd):
    """
    Analytically solve for the optimal weight
    :param data: dict, X,t
    :param lambd: float, regularization factor
    :return: np.ndarray, a weight vector
    """
    X = data['X']
    N = X.shape[0]
    K = X.shape[1]
    t = data['t']
    return sla.solve(X.T @ X + N * lambd * np.eye(K), X.T @ t)


def predict(data, model):
    """
    Return predicted target based on data and weight parameters
    :param data: dict, X,t
    :param model: np.ndarray, weight vector
    :return: np.ndarray, predicted variable
    """
    return data['X'] @ model

def loss(data, model):
    """
    Calculate the loss from prediction VS actual data
    :param data: dict, X,t
    :param model: np.ndarray, weight vector
    :return: float, total cost
    """
    X = data['X']
    t = data['t']
    N = X.shape[0]
    predictions = predict(data, model)
    return np.sum(np.power(predictions - t, 2))/2/N

def cross_validation(data, num_folds, lambd_seq):
    """
    Cross Validation
    :param data: dict, train and validation data with X,t
    :param num_folds: int, number of folds to divide the data by
    :param lambd_seq: list/np.ndarray, vector of lambd values
    :return: np.ndarray, cross validation error given lambd
    """
    data = shuffle_data(data)
    cv_error = np.zeros(len(lambd_seq))
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error[i] = cv_loss_lmd/num_folds
    return cv_error

def calc_err(data_train, data_test, lambd_seq):
    """
    Calculate losses without cross validation
    :param data_train: dict, train data with X,t
    :param data_test: dict, test data with X,t
    :param lambd_seq: list/np.ndarray, vector of lambd values
    :return: tuple[np.ndarray], train error and test error
    """
    train_error = np.zeros(len(lambd_seq))
    test_error = np.zeros(len(lambd_seq))
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        model = train_model(data_train, lambd)
        train_error[i] = loss(data_train, model)
        test_error[i] = loss(data_test, model)
    return train_error, test_error


if __name__ == "__main__":

    lambd_seq = np.linspace(0.00005, 0.005, 50)

    # Without Cross Validation
    train_err, test_err = calc_err(data_train, data_test, lambd_seq)
    plt.plot(lambd_seq, train_err, label="Train Error");
    plt.plot(lambd_seq, test_err, label="Test Error");
    plt.legend();
    plt.title("Train VS Test Error");
    plt.xlabel("lambd");
    plt.ylabel("error");
    plt.savefig('q4_train_test.png');
    plt.show();

    # With Cross Validation
    cv_err_five_fold = cross_validation(data_train, 5, lambd_seq)
    cv_err_ten_fold = cross_validation(data_train, 10, lambd_seq)
    plt.plot(lambd_seq, train_err, label="Train Error");
    plt.plot(lambd_seq, test_err, label="Test Error");
    plt.plot(lambd_seq, cv_err_five_fold, label="5-fold CV Error");
    plt.plot(lambd_seq, cv_err_ten_fold, label="10-fold CV Error");
    plt.legend();
    plt.title("Train VS Test Error");
    plt.xlabel("lambd");
    plt.ylabel("error");
    plt.savefig('q4_cv.png');
    plt.show();

    print("Optimal Lambda with 5-fold CV:", lambd_seq[cv_err_five_fold.argmin()])
    print("Optimal Lambda with 10-fold CV:", lambd_seq[cv_err_ten_fold.argmin()])