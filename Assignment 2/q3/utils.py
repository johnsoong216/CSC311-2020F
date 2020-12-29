import numpy as np
import matplotlib.pyplot as plt


def load_data(file_name):
    """ Loads the data.
    """
    npzfile = np.load(file_name)

    inputs_train = npzfile["inputs_train"].T / 255.0
    inputs_valid = npzfile["inputs_valid"].T / 255.0
    inputs_test = npzfile["inputs_test"].T / 255.0
    target_train = npzfile["target_train"].tolist()
    target_valid = npzfile["target_valid"].tolist()
    target_test = npzfile["target_test"].tolist()

    num_class = max(target_train + target_valid + target_test) + 1
    target_train_1hot = np.zeros([num_class, len(target_train)])
    target_valid_1hot = np.zeros([num_class, len(target_valid)])
    target_test_1hot = np.zeros([num_class, len(target_test)])

    for ii, xx in enumerate(target_train):
        target_train_1hot[xx, ii] = 1.0

    for ii, xx in enumerate(target_valid):
        target_valid_1hot[xx, ii] = 1.0

    for ii, xx in enumerate(target_test):
        target_test_1hot[xx, ii] = 1.0

    inputs_train = inputs_train.T
    inputs_valid = inputs_valid.T
    inputs_test = inputs_test.T
    target_train_1hot = target_train_1hot.T
    target_valid_1hot = target_valid_1hot.T
    target_test_1hot = target_test_1hot.T
    return inputs_train, inputs_valid, inputs_test, target_train_1hot, target_valid_1hot, target_test_1hot


def save(file_name, data):
    """ Saves the model to a numpy file.
    """
    print("Writing to " + file_name)
    np.savez_compressed(file_name, data)


def load(file_name):
    """ Loads the model from numpy file.
    """
    print("Loading from " + file_name)
    return dict(np.load(file_name, allow_pickle=True))


def display_plot(train, valid, y_label, alpha, batch_size, number=0, above=1, threshold=0):
    """ Displays training curve.
    :param train: Training statistics
    :param valid: Validation statistics
    :param y_label: Y-axis label of the plot
    :param number: The number of the plot
    :return: None
    """
    plt.figure(number)
    plt.clf()
    train = np.array(train)
    valid = np.array(valid)

    train_label = {0: "Train", 1: "Train Above " + str(threshold), -1: "Validation Above" + str(threshold)}
    validation_label = {0: "Validation", 1: "Train Below" + str(threshold), -1: "Validation Below" + str(threshold)}

    plt.plot(train[:, 0], train[:, 1], "b", label=train_label[above])
    plt.plot(valid[:, 0], valid[:, 1], "g", label=validation_label[above])
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(f"Alpha: {alpha} Batch Size: {batch_size // 10 * 10}");
    plt.legend()
    plt.draw()
    plt.savefig(f"{y_label}_alpha_{alpha}_batch_size_{batch_size}_{train_label[above].split(' ')[0] + str(threshold) if above != 0 else ''}.png")
    # plt.pause(0.01)

