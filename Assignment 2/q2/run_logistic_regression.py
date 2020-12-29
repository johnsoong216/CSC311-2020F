from q2.check_grad import check_grad
from q2.utils import *
from q2.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()



    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 100
    }
    weights = np.zeros(M+1).reshape(-1, 1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################

    # The for loop is set up to try different learning rate/iteration values
    learning_rate_list = [0.1] # [0.001, 0.01, 0.1, 1.0]
    num_iterations_list = [250] # [50, 100, 250, 500]
    for learning_rate in learning_rate_list:
        for num_iterations in num_iterations_list:
            train_loss = []
            validation_loss = []
            iteration = []

            # Set Learning Rate and Number of Iterations
            hyperparameters["learning_rate"] = learning_rate
            hyperparameters["num_iterations"] = num_iterations

            # Initialize Weight
            # weights = np.random.randn(M + 1).reshape(-1, 1)
            weights = np.zeros(M+1).reshape(-1, 1)
            for t in range(hyperparameters["num_iterations"]):
                loss, grad, pred = logistic(weights, train_inputs, train_targets, hyperparameters)
                weights = weights - hyperparameters["learning_rate"] * grad

                train_loss.append(evaluate(train_targets, y=logistic_predict(weights, train_inputs))[0])
                validation_loss.append(evaluate(valid_targets, y=logistic_predict(weights, valid_inputs))[0])
                iteration.append(t)

            plt.plot(iteration, train_loss, label="train loss", c='b');
            plt.plot(iteration, validation_loss, label="validation loss", c='g');
            plt.xlabel("Iteration");
            plt.ylabel("Cross Entropy");
            plt.legend();
            plt.title(f"Alpha: {learning_rate} Iterations: {num_iterations}");
            plt.savefig(f"q22_iterations_{num_iterations}_alpha_{learning_rate}.png");
            plt.show()

        print("Train: ", evaluate(train_targets, logistic_predict(weights, train_inputs)))
        print("Validation:", evaluate(valid_targets, logistic_predict(weights, valid_inputs)))
        print("Test: ", evaluate(test_targets, logistic_predict(weights, test_inputs)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_pen_logistic_regression():
    train_inputs, train_targets = load_train()

    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()


    N, M = train_inputs.shape


    #####################################################################
    # TODO:                                                             #
    # Implement the function that automatically evaluates different     #
    # penalty and re-runs penalized logistic regression 5 times.        #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        "weight_regularization": 0.,
        "num_iterations": 250
    }

    # Number of Runs and Weights to Run
    num_rerun = 5
    weight_lambd = [0, 0.001, 0.01, 0.1, 1.0]


    for lambd in weight_lambd:

        # Initialize Variables
        train_loss = [0] * hyperparameters["num_iterations"]
        validation_loss = [0] * hyperparameters["num_iterations"]
        iteration = [0] * hyperparameters["num_iterations"]

        train_ce_total = 0
        train_acc_total = 0
        valid_ce_total = 0
        valid_acc_total = 0

        # Run Multiple Times
        for i in range(num_rerun):
            # weights = np.zeros(M + 1).reshape(-1, 1)

            # Use Random Weight
            weights = np.random.randn(M + 1).reshape(-1, 1) * 0.001
            hyperparameters["weight_regularization"] = lambd

            for t in range(hyperparameters["num_iterations"]):
                loss, grad, pred = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                weights -= hyperparameters["learning_rate"] * grad

                train_loss[t] += evaluate(train_targets, y=logistic_predict(weights, train_inputs))[0]/num_rerun
                validation_loss[t] += evaluate(valid_targets, y=logistic_predict(weights, valid_inputs))[0]/num_rerun
                iteration[t] += t/num_rerun

            # Calculate CE and Acc for Individual Run
            train_ce, train_acc = evaluate(train_targets, logistic_predict(weights, train_inputs))
            valid_ce, valid_acc = evaluate(valid_targets, logistic_predict(weights, valid_inputs))

            # Add to the total Run
            train_ce_total += train_ce/num_rerun
            train_acc_total += train_acc/num_rerun
            valid_ce_total += valid_ce/num_rerun
            valid_acc_total += valid_acc/num_rerun

        # Print CE and Accuracy
        print("Lambda: ", lambd)
        print("Train CE: ", train_ce_total, " Accuracy: " , train_acc_total, " Error: ", 1 - train_acc_total)
        print("Valid CE: ", valid_ce_total, " Accuracy: " , valid_acc_total, " Error: ", 1 - valid_acc_total)

        # Plot the Losses
        plt.plot(iteration, train_loss, label="train loss");
        plt.plot(iteration, validation_loss, label="validation loss");
        plt.xlabel("Iteration");
        plt.ylabel("Cross Entropy");
        plt.legend();
        plt.title(f"Regularization Lambd: {lambd}");
        plt.savefig(f"q23_lambd_{lambd}_train.png");
        plt.show()

        # Print the Test Result for Q2.3
        if lambd == 0.1:
            test_ce, test_acc = evaluate(test_targets, logistic_predict(weights, test_inputs))
            print("Test CE: ", test_ce, " Accuracy: " , test_acc, "Error: ", 1 - test_acc)



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    run_pen_logistic_regression()
