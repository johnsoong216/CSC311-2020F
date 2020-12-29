""" Instruction:

In this section, you are asked to train a NN with different hyperparameters.
To start with training, you need to fill in the incomplete code. There are 3
places that you need to complete:
a) Backward pass equations for an affine layer (linear transformation + bias).
b) Backward pass equations for ReLU activation function.
c) Weight update equations.

After correctly fill in the code, modify the hyperparameters in "main()".
You can then run this file with the command: "python nn.py" in your terminal.
The program will automatically check your gradient implementation before start.
The program will print out the training progr ess, and it will display the
training curve by the end. You can optionally save the model by uncommenting
the lines in "main()".
"""


from q3.utils import load_data, load, save, display_plot
import numpy as np
import matplotlib.pyplot as plt


def init_nn(num_inputs, num_hiddens, num_outputs):
    """ Initializes neural network's parameters.
    :param num_inputs: Number of input units
    :param num_hiddens: List of two elements, hidden size for each layers.
    :param num_outputs: Number of output units
    :return: A dictionary of randomly initialized neural network weights.
    """

    # print("Inputs: ", num_inputs)
    # print("Hiddens: ", num_hiddens)
    W1 = 0.1 * np.random.randn(num_inputs, num_hiddens[0])
    W2 = 0.1 * np.random.randn(num_hiddens[0], num_hiddens[1])
    W3 = 0.01 * np.random.randn(num_hiddens[1], num_outputs)
    b1 = np.zeros((num_hiddens[0]))
    b2 = np.zeros((num_hiddens[1]))
    b3 = np.zeros((num_outputs))
    model = {
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "b1": b1,
        "b2": b2,
        "b3": b3
    }

    return model


def affine(x, w, b):
    """ Computes the affine transformation.
    :param x: Inputs (or hidden layers)
    :param w: Weight
    :param b: Bias
    :return: Outputs
    """
    y = x.dot(w) + b
    return y


def affine_backward(grad_y, x, w):
    """ Computes gradients of affine transformation.
    Hint: you may need the matrix transpose np.dot(A, B).T = np.dot(B, A) and (A.T).T = A
    :param grad_y: Gradient from upper layer
    :param x: Inputs from the hidden layer
    :param w: Weights
    :return: A tuple of (grad_h, grad_w, grad_b)
        WHERE
        grad_x: Gradients wrt. the inputs/hidden layer.
        grad_w: Gradients wrt. the weights.
        grad_b: Gradients wrt. the biases.
    """
    #####################################################################
    # TODO:                                                             #
    # Complete the function to compute the gradients of affine          #
    # transformation.                                                   #
    #####################################################################


    grad_x = grad_y @ w.T
    grad_w = x.T @ grad_y #(grad_y.T @ x).T
    grad_b = np.sum(grad_y, axis=0)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return grad_x, grad_w, grad_b


def relu(x):
    """ Computes the ReLU activation function.
    :param z: Inputs
    :return: Activation of x
    """
    return np.maximum(x, 0.0)


def relu_backward(grad_y, x):
    """ Computes gradients of the ReLU activation function wrt. the unactivated inputs.
    :param grad_y: Gradient of the activation.
    :param x: Inputs
    :return: Gradient wrt. x
    """
    #####################################################################
    # TODO:                                                             #
    # Complete the function to compute the gradients of relu.           #
    #####################################################################

    grad_x = np.array(x > 0, dtype=int) * grad_y

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return grad_x


def softmax(x):
    """ Computes the softmax activation function.
    :param x: Inputs
    :return: Activation of x
    """
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


def nn_forward(model, x):
    """ Runs a forward pass.
    :param model: Dictionary of all the weights.
    :param x: Input to the network.
    :return: Dictionary of all intermediate variables.
    """
    z1 = affine(x, model["W1"], model["b1"])
    h1 = relu(z1)
    z2 = affine(h1, model["W2"], model["b2"])
    h2 = relu(z2)
    y = affine(h2, model["W3"], model["b3"])
    var = {
        "x": x,
        "z1": z1,
        "h1": h1,
        "z2": z2,
        "h2": h2,
        "y": y
    }
    return var


def nn_backward(model, err, var):
    """ Runs the backward pass.
    :param model: Dictionary of all the weights.
    :param err: Gradients to the output of the network.
    :param var: Intermediate variables from the forward pass.
    :return: None
    """
    dE_dh2, dE_dW3, dE_db3 = affine_backward(err, var["h2"], model["W3"])

    dE_dz2 = relu_backward(dE_dh2, var["z2"])

    dE_dh1, dE_dW2, dE_db2 = affine_backward(dE_dz2, var["h1"], model["W2"])

    dE_dz1 = relu_backward(dE_dh1, var["z1"])

    _, dE_dW1, dE_db1 = affine_backward(dE_dz1, var["x"], model["W1"])

    model["dE_dW1"] = dE_dW1
    model["dE_dW2"] = dE_dW2
    model["dE_dW3"] = dE_dW3
    model["dE_db1"] = dE_db1
    model["dE_db2"] = dE_db2
    model["dE_db3"] = dE_db3
    return


def nn_update(model, alpha):
    """ Update NN weights.
    :param model: Dictionary of all the weights.
    :param alpha: Learning rate
    :return: None
    """
    #####################################################################
    # TODO:                                                             #
    # Complete the function to update the neural network's parameters.  #
    # Your code should look as follows                                  #
    # model["W1"] = ...                                                 #
    # model["W2"] = ...                                                 #
    # ...                                                               #
    #####################################################################
    model["W1"] = model["W1"] - alpha * model["dE_dW1"]
    model["W2"] = model["W2"] - alpha * model["dE_dW2"]
    model["W3"] = model["W3"] - alpha * model["dE_dW3"]
    model["b1"] = model["b1"] - alpha * model["dE_db1"]
    model["b2"] = model["b2"] - alpha * model["dE_db2"]
    model["b3"] = model["b3"] - alpha * model["dE_db3"]

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return


def train(model, forward, backward, update, alpha, num_epochs, batch_size, threshold=0.5):
    """ Trains a simple MLP.
    :param model: Dictionary of model weights.
    :param forward: Forward prop function.
    :param backward: Backward prop function.
    :param update: Update weights function.
    :param alpha: Learning rate.
    :param num_epochs: Number of epochs to run training for.
    :param batch_size: Mini-batch size, -1 for full batch.
    :return: A tuple (train_ce, valid_ce, train_acc, valid_acc)
        WHERE
        train_ce: Training cross entropy.
        valid_ce: Validation cross entropy.
        train_acc: Training accuracy.
        valid_acc: Validation accuracy.
    """
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, \
        target_test = load_data("data/toronto_face.npz")
    rnd_idx = np.arange(inputs_train.shape[0])
    
    train_ce_list = []
    valid_ce_list = []
    train_acc_list = []
    valid_acc_list = []
    train_acc_above_list = []
    train_acc_below_list = []
    valid_acc_above_list = []
    valid_acc_below_list = []

    num_train_cases = inputs_train.shape[0]
    if batch_size == -1:
        batch_size = num_train_cases
    num_steps = int(np.ceil(num_train_cases / batch_size))
    for epoch in range(num_epochs):
        np.random.shuffle(rnd_idx)
        inputs_train = inputs_train[rnd_idx]
        target_train = target_train[rnd_idx]
        for step in range(num_steps):
            # Forward pass.
            start = step * batch_size
            end = min(num_train_cases, (step + 1) * batch_size)
            x = inputs_train[start: end]
            t = target_train[start: end]

            var = forward(model, x)
            prediction = softmax(var["y"])

            train_ce = -np.sum(t * np.log(prediction)) / float(x.shape[0])
            train_acc = (np.argmax(prediction, axis=1) ==
                         np.argmax(t, axis=1)).astype("float").mean()

            print(("Epoch {:3d} Step {:2d} Train CE {:.5f} "
                   "Train Acc {:.5f}").format(
                epoch, step, train_ce, train_acc))

            # Compute error.
            error = (prediction - t) / float(x.shape[0])

            # Backward prop.
            backward(model, error, var)

            # Update weights.
            update(model, alpha)

        valid_ce, valid_acc = evaluate(
            inputs_valid, target_valid, model, forward, batch_size=batch_size)
        print(("Epoch {:3d} "
               "Validation CE {:.5f} "
               "Validation Acc {:.5f}\n").format(
            epoch, valid_ce, valid_acc))


        train_ce_list.append((epoch, train_ce))
        train_acc_list.append((epoch, train_acc))
        valid_ce_list.append((epoch, valid_ce))
        valid_acc_list.append((epoch, valid_acc))


    display_plot(train_ce_list, valid_ce_list, "Cross Entropy", alpha=alpha, batch_size=batch_size, number=0, above=0)
    display_plot(train_acc_list, valid_acc_list, "Accuracy", alpha=alpha, batch_size=batch_size, number=1, above=0)
    # display_plot(train_acc_above_list, train_acc_below_list, "Accuracy", alpha=alpha, batch_size=batch_size, number=2, above=1, threshold=threshold)
    # display_plot(valid_acc_above_list, valid_acc_below_list, "Accuracy", alpha=alpha, batch_size=batch_size, number=3, above=-1, threshold=threshold)



    train_ce, train_acc = evaluate(
        inputs_train, target_train, model, forward, batch_size=batch_size)
    valid_ce, valid_acc = evaluate(
        inputs_valid, target_valid, model, forward, batch_size=batch_size)
    test_ce, test_acc = evaluate(
        inputs_test, target_test, model, forward, batch_size=batch_size)
    print("CE: Train %.5f Validation %.5f Test %.5f" %
          (train_ce, valid_ce, test_ce))
    print("Acc: Train {:.5f} Validation {:.5f} Test {:.5f}".format(
        train_acc, valid_acc, test_acc))

    stats = {
        "train_ce": train_ce_list,
        "valid_ce": valid_ce_list,
        "train_acc": train_acc_list,
        "valid_acc": valid_acc_list
    }

    return model, stats


def evaluate(inputs, target, model, forward, batch_size=-1):
    """ Evaluates the model on inputs and target.
    :param inputs: Inputs to the network
    :param target: Target of the inputs
    :param model: Dictionary of network weights
    :param forward: Function for forward pass
    :param batch_size: Batch size
    :return: A tuple (ce, acc)
        WHERE
        ce: cross entropy
        acc: accuracy
    """
    num_cases = inputs.shape[0]
    if batch_size == -1:
        batch_size = num_cases
    num_steps = int(np.ceil(num_cases / batch_size))
    ce = 0.0
    acc = 0.0

    for step in range(num_steps):
        start = step * batch_size
        end = min(num_cases, (step + 1) * batch_size)
        x = inputs[start: end]
        t = target[start: end]
        prediction = softmax(forward(model, x)["y"])
        ce += -np.sum(t * np.log(prediction))
        acc += (np.argmax(prediction, axis=1) == np.argmax(
            t, axis=1)).astype("float").sum()

    ce /= num_cases
    acc /= num_cases

    return ce, acc


def threshold_evaluate(inputs, target, model, forward, batch_size=-1, threshold=0.5):
    """
    Evaluates cross entropy and accuracy for top score classes below/above a certain threshold
    """
    num_cases = inputs.shape[0]
    if batch_size == -1:
        batch_size = num_cases
    num_steps = int(np.ceil(num_cases / batch_size))
    ce = 0.0
    acc = 0.0
    acc_above = 0.0
    num_cases_above = 0.0
    acc_below = 0.0
    num_cases_below = 0.0
    for step in range(num_steps):
        start = step * batch_size
        end = min(num_cases, (step + 1) * batch_size)
        x = inputs[start: end]
        t = target[start: end]
        prediction = softmax(forward(model, x)["y"])
        ce += -np.sum(t * np.log(prediction))
        acc += (np.argmax(prediction, axis=1) == np.argmax(
            t, axis=1)).astype("float").sum()

        predict_max = np.max(prediction, axis=1)
        acc_above += ((np.argmax(prediction, axis=1)[predict_max > threshold]) == (
        np.argmax(t, axis=1)[predict_max > threshold])
                                     ).astype("float").sum()
        num_cases_above += (predict_max > threshold).sum()
        acc_below += ((np.argmax(prediction, axis=1)[predict_max <= threshold]) == (
        np.argmax(t, axis=1)[predict_max <= threshold])
                                     ).astype("float").sum()
        num_cases_below += (predict_max <= threshold).sum()


    ce /= num_cases

    print("Above: ", acc_above, " Num Cases: ", num_cases_above)
    print("Below: ", acc_below, " Num Cases: ", num_cases_below)
    print("Overall: ", acc, " Num cases: ", num_cases)

    acc /= num_cases
    acc_above /= num_cases_above
    acc_below /= num_cases_below

    return ce, acc, acc_above, acc_below, num_cases, num_cases_above, num_cases_below


def individual_evaluate(inputs, target, model, forward, batch_size=-1, threshold=0.5):
    """
    Plots all the images with top score class below a given threshold
    """

    emotion_dict = {1: "Anger", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Sad", 6: "Surprise", 7:" Neutral"}
    num_cases = inputs.shape[0]
    if batch_size == -1:
        batch_size = num_cases
    num_steps = int(np.ceil(num_cases / batch_size))
    below_threshold = []
    for step in range(num_steps):
        start = step * batch_size
        end = min(num_cases, (step + 1) * batch_size)
        x = inputs[start: end]
        prediction = softmax(forward(model, x)["y"])
        if np.max(prediction) <= threshold:
            below_threshold.append(step)

    size = int(np.ceil(np.sqrt(len(below_threshold))))
    fig, ax = plt.subplots(ncols=size, nrows=size, figsize=(28,15))
    for step, x in enumerate(inputs[below_threshold]):
        ax[step // size][step % size].imshow(x.reshape(48, -1), cmap="gray");
        prediction = softmax(forward(model, x[np.newaxis, :])["y"])
        t = target[step:step+1]

        ax[step // size][step % size].set_title(f"Prediction: {emotion_dict[np.argmax(prediction) + 1]} Actual: {emotion_dict[np.argmax(t) + 1]}")
        ax[step // size][step % size].get_xaxis().set_visible(False);
        ax[step // size][step % size].get_yaxis().set_visible(False);
    fig.savefig(f"Below{threshold} Facial Expressions.png");

    for step in range(len(below_threshold), size**2):
        ax[step // size][step % size].axis('off');
        ax[step // size][step % size].get_xaxis().set_visible(False);
        ax[step // size][step % size].get_yaxis().set_visible(False);




def check_grad(model, forward, backward, name, x):
    """ Check the gradients.
    """
    np.random.seed(0)
    var = forward(model, x)
    loss = lambda y: 0.5 * (y ** 2).sum()
    grad_y = var["y"]
    backward(model, grad_y, var)
    grad_w = model["dE_d" + name].ravel()
    w_ = model[name].ravel()
    eps = 1e-7
    grad_w_2 = np.zeros(w_.shape)
    check_elem = np.arange(w_.size)
    np.random.shuffle(check_elem)
    # Randomly check 20 elements.
    check_elem = check_elem[:20]
    for ii in check_elem:
        w_[ii] += eps
        err_plus = loss(forward(model, x)["y"])
        w_[ii] -= 2 * eps
        err_minus = loss(forward(model, x)["y"])
        w_[ii] += eps
        grad_w_2[ii] = (err_plus - err_minus) / 2. / eps
    np.testing.assert_almost_equal(grad_w[check_elem], grad_w_2[check_elem],
                                   decimal=3)


def main():
    """ Trains a neural network.
    :return: None
    """
    model_file_name = "nn_model.npz"
    stats_file_name = "nn_stats.npz"
    hypertune_file_name = "nn_hypertune.npz"

    # Hyper-parameters. Modify them if needed.
    num_hiddens = [16, 32]
    alpha = 0.01
    num_epochs = 1000
    batch_size = 100
    threshold = 0.8

    # Input-output dimensions.
    num_inputs = 2304
    num_outputs = 7

    # Initialize model.
    model = init_nn(num_inputs, num_hiddens, num_outputs)

    # Uncomment to reload trained model here.
    # model = load(model_file_name)

    # Check gradient implementation.
    print("Checking gradients...")
    x = np.random.rand(10, 48 * 48) * 0.1
    check_grad(model, nn_forward, nn_backward, "W3", x)
    check_grad(model, nn_forward, nn_backward, "b3", x)
    check_grad(model, nn_forward, nn_backward, "W2", x)
    check_grad(model, nn_forward, nn_backward, "b2", x)
    check_grad(model, nn_forward, nn_backward, "W1", x)
    check_grad(model, nn_forward, nn_backward, "b1", x)

    # Train model.
    stats = train(model, nn_forward, nn_backward, nn_update, alpha,
                  num_epochs, batch_size, threshold)


    # Q3C
    # alpha_list = [0.001, 0.01, 0.1, 0.5, 1.0]
    # batch_size_list = [10, 100, 250, 500, 1000]
    #
    # hypertune = {}
    # for alpha in alpha_list:
    #     for batch_size in batch_size_list:
    #         model = init_nn(num_inputs, num_hiddens, num_outputs)
    #         stats = train(model, nn_forward, nn_backward, nn_update, alpha,
    #               num_epochs, batch_size, threshold)
    #     hypertune[(alpha, batch_size)] = stats
    # save(hypertune_file_name, hypertune)


    # Q3D
    # num_hiddens_list = [[32, 4], [32, 16], [32, 64], [16, 4], [16, 16], [16, 64], [64, 4], [64,16], [64,64]]
    # for idx, num_hiddens in enumerate(num_hiddens_list):
    #     alpha = 0.01
    #     num_epochs = 1000
    #     batch_size = 100
    #     threshold = 0.8
    #
    #     # Input-output dimensions.
    #     num_inputs = 2304
    #     num_outputs = 7
    #
    #     # Initialize model.
    #     model = init_nn(num_inputs, num_hiddens, num_outputs)
    #     stats = train(model, nn_forward, nn_backward, nn_update, alpha,
    #                   num_epochs, batch_size, threshold)


    # Q3E Code
    # acc_list = []
    # acc_above_list = []
    # acc_below_list = []
    # total_sample_list = []
    # above_sample_list = []
    # below_sample_list = []
    # threshold_list = list(np.linspace(0.5, 0.95, 19))
    # dataset_list = [(inputs_train, target_train), (inputs_valid, target_valid), (inputs_test, target_test)]
    #
    # for dataset in dataset_list:
    #     for threshold in threshold_list:
    #         ce, acc, acc_above, acc_below, total_sample, above_sample, below_sample = threshold_evaluate(dataset[0],
    #                                                                                                      dataset[1],
    #                                                                                                      model,
    #                                                                                                      nn_forward,
    #                                                                                                      batch_size=100,
    #                                                                                                      threshold=threshold)
    #         acc_list.append(acc)
    #         acc_above_list.append(acc_above)
    #         acc_below_list.append(acc_below)
    #         total_sample_list.append(total_sample)
    #         above_sample_list.append(above_sample)
    #         below_sample_list.append(below_sample)

    # summary_df = pd.DataFrame(np.array(
    #     [acc_list, acc_above_list, acc_below_list, total_sample_list, above_sample_list, below_sample_list,
    #      ["train"] * len(threshold_list) + ["valid"] * len(threshold_list) + ["test"] * len(threshold_list),
    #      threshold_list * 3]).T,
    #                           columns=["ACCURACY", "ACCURACY_ABOVE_THRESHOLD", "ACCURACY_BELOW_THRESHOLD", "SAMPLE",
    #                                    "SAMPLE_ABOVE_THRESHOLD",
    #                                    "SAMPLE_BELOW_THRESHOLD", "DATA_TYPE", "THRESHOLD"])

    # summary_df = summary_df.apply(pd.to_numeric, errors="ignore")

    # plt.plot(summary_df.query("DATA_TYPE == 'train'")["THRESHOLD"],
    #          summary_df.query("DATA_TYPE == 'train'")["ACCURACY_BELOW_THRESHOLD"], color="r", marker="v",
    #          label="Train Below Threshold");
    # plt.plot(summary_df.query("DATA_TYPE == 'train'")["THRESHOLD"],
    #          summary_df.query("DATA_TYPE == 'train'")["ACCURACY_ABOVE_THRESHOLD"], color="r", marker="o",
    #          label="Train Above Threshold");
    # plt.plot(summary_df.query("DATA_TYPE == 'valid'")["THRESHOLD"],
    #          summary_df.query("DATA_TYPE == 'valid'")["ACCURACY_BELOW_THRESHOLD"], color="g", marker="v",
    #          label="Valid Below Threshold");
    # plt.plot(summary_df.query("DATA_TYPE == 'valid'")["THRESHOLD"],
    #          summary_df.query("DATA_TYPE == 'valid'")["ACCURACY_ABOVE_THRESHOLD"], color="g", marker="o",
    #          label="Valid Above Threshold");
    # plt.plot(summary_df.query("DATA_TYPE == 'test'")["THRESHOLD"],
    #          summary_df.query("DATA_TYPE == 'test'")["ACCURACY_BELOW_THRESHOLD"], color="b", marker="v",
    #          label="Test Below Threshold");
    # plt.plot(summary_df.query("DATA_TYPE == 'test'")["THRESHOLD"],
    #          summary_df.query("DATA_TYPE == 'test'")["ACCURACY_ABOVE_THRESHOLD"], color="b", marker="o",
    #          label="Test Above Threshold");
    # plt.legend(bbox_to_anchor=(1.1, 1.05));
    # plt.xlabel("Threshold");
    # plt.ylabel("Accuracy");
    # plt.title("Batch Size = 100, Alpha = 0.01, Threshold VS Accuracy");
    # plt.show()
    # individual_evaluate(inputs_test, target_test, model, nn_forward, batch_size=1, threshold=0.5)


    # Uncomment if you wish to save the model.
    # save(model_file_name, model)

    # Uncomment if you wish to save the training statistics.
    # save(stats_file_name, stats)


if __name__ == "__main__":
    main()
