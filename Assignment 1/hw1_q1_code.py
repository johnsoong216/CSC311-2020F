import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data_path = "hw1_data/"

def load_data(real_fp="clean_real.txt", fake_fp="clean_fake.txt", train_size=0.7, val_size=0.15):
    """
    Function to load the Real/Fake News data
    :param real_fp: str, file path of real data
    :param fake_fp: str, file path of fake data
    :param train_size: float, proportion of the train set
    :param val_size: float, proportion of the validation set
    :return: tuple of train, val, test set (Data and Label)
    """

    # Initialize a CountVectorizer
    cv = CountVectorizer()

    # Read Data Input
    data = []
    f = open(data_path + real_fp)
    for line in f:
        data.append(line.strip())
    real_length = len(data)
    f.close()
    f = open(data_path + fake_fp)
    for line in f:
        data.append(line.strip())

    # Transform Data into Matrix
    data_arr = cv.fit_transform(data)
    # Create Label
    label = np.concatenate([np.ones(real_length), np.zeros(len(data) - real_length)])

    # Split Data
    train_data, val_test_data, train_label, val_test_label = train_test_split(data_arr, label, train_size=train_size)
    val_data, test_data, val_label, test_label = train_test_split(val_test_data, val_test_label, train_size=val_size/(1-train_size))
    return train_data, val_data, test_data, train_label, val_label, test_label


def select_knn_model(train_data, val_data, test_data, train_label, val_label, test_label, dim_range, **kwargs):
    """
    Function to hypertune k (dimension) in the KNN model and output the optimal K and the test accuracy
    :param train_data: train data
    :param val_data: validation data
    :param test_data: test data
    :param train_label: train label
    :param val_label: validation label
    :param test_label: test label
    :param dim_range: int, maximum dimension to hypertune
    :param kwargs: parameters to pass into KNN
    :return: tuple of the optimal dimension, and the test accuracy
    """

    # Initialize an array of training/validation accuracy
    train_acc = np.zeros(dim_range)
    val_acc = np.zeros(dim_range)

    # Iterate through all possible dimensions
    for i, k in enumerate(np.arange(1, dim_range + 1)):

        # Initialize the Model
        knn_model = KNN(n_neighbors=k, **kwargs)

        # Fit the Model
        knn_model.fit(train_data, train_label)

        # Record training and validation accuracy
        train_acc[i] = (knn_model.predict(train_data) == train_label).sum() / len(train_label)
        val_acc[i] = (knn_model.predict(val_data) == val_label).sum() / len(val_label)


    # Plot a chart of training and validation accuracy
    plt.plot(np.arange(1, dim_range + 1), train_acc, label="train set accuracy");
    plt.plot(np.arange(1, dim_range + 1), val_acc, label="validation set accuracy");
    plt.xlabel("Dimension");
    plt.xticks(np.arange(0, dim_range + 1));
    plt.ylabel("Accuracy");
    plt.title("Model Accuracy VS Dimension");
    plt.legend();
    if "metric" in kwargs:
        plt.savefig('q1_metric.png');
    else:
        plt.savefig('q1.png');
    plt.show();

    # Find the optimal Dimension and apply on the test set
    optimal_dim = val_acc.argmax() + 1
    final_model = KNN(n_neighbors=optimal_dim, **kwargs)
    final_model.fit(train_data, train_label)
    test_acc = (final_model.predict(test_data) == test_label).sum() / len(test_label)

    return optimal_dim, test_acc


if __name__ == "__main__":
    train_data, val_data, test_data, train_label, val_label, test_label = load_data()
    opt_dim, test_acc = select_knn_model(train_data, val_data, test_data, train_label, val_label, test_label, 20)
    print("Optimal Dimension: ", opt_dim, " Test Accuracy: ", test_acc)
    opt_dim, test_acc = select_knn_model(train_data, val_data, test_data, train_label, val_label, test_label, 20, metric='cosine')
    print("Optimal Dimension: ", opt_dim, " Test Accuracy: ", test_acc)
