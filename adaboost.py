"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np
import math
import ex4_tools
import matplotlib.pyplot as plt


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights
        self.vfunc = np.vectorize(self.WL.predict, otypes=[np.ndarray],
                                  excluded=['X'])
        self.Dt = None

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last
        iteration.
        """
        Dt = np.full([y.shape[0]], 1 / y.shape[0])  # uniform distribution
        for i in range(self.T):
            ht = self.WL(Dt, X, y)
            self.h[i] = ht
            yt = ht.predict(X)
            eps_t = np.sum((((y + yt) == 0) * Dt))
            wt = 0.5 * math.log((1 / eps_t) - 1)
            self.w[i] = wt
            Dt = Dt * np.exp(-wt * y * yt)
            Dt = Dt / np.sum(Dt)
        self.Dt = Dt
        return Dt

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for
        the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        return np.sign(
            np.sum(self.w[:max_t] * self.vfunc(self.h[:max_t], X=X)))

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for
        the classification
        :return: error : the ratio of the wrong predictions when predict only
        with max_t weak learners (float)
        """
        y_hat = self.predict(X, max_t)
        return np.count_nonzero(y - y_hat) / y.shape[0]


def train_and_plot(noise_ratio):
    """
    Train the Adaboost Classifier, and plot error rates and decision
    boundaries.
    :param noise_ratio: The noise level to generate the data for the Adaboost
    Classifier to train and test with.
    """
    X_train, y_train = ex4_tools.generate_data(num_of_train_samples,
                                               noise_ratio)
    X_test, y_test = ex4_tools.generate_data(num_of_test_samples, noise_ratio)
    train_error = np.zeros(T)
    test_error = np.zeros(T)

    adaboost = AdaBoost(ex4_tools.DecisionStump, T)
    adaboost.train(X_train, y_train)

    for t in range(1, T + 1):
        if t in decision_boundaries_T:
            ex4_tools.decision_boundaries(adaboost, X_test, y_test, t)
            plt.savefig(f'Plots/boundaries_{t}T_{int(noise_ratio*100)}noise')
        train_error[t - 1] = adaboost.error(X_train, y_train, t)
        test_error[t - 1] = adaboost.error(X_test, y_test, t)

    plot_error_rates(train_error, test_error, noise_ratio)

    plot_best_T_boundaries(adaboost, X_train, y_train, test_error, noise_ratio)

    plot_weighted_boundaries(adaboost, X_train, y_train, noise_ratio)


def plot_error_rates(train_error, test_error, noise_ratio):
    """
    Plot the train and test error rates provided as a function of the number
    of Adaboost iterations.
    :param noise_ratio: The noise ratio the data was generated with.
    :param train_error: The train error rates data.
    :param test_error: The train error rates data.
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    x = np.arange(1, T + 1)
    ax.plot(x, train_error, label="Train Error")
    ax.plot(x, test_error, label="Test Error")
    ax.legend()
    ax.set_title(f"Training\\Test Errors of Data with {noise_ratio} "
                 f"Noise Ratio by # of Classifiers")
    ax.set_xlabel("# of Classifiers")
    ax.set_ylabel("Error Rate")
    plt.savefig(f'Plots/train_test_error_{int(noise_ratio*100)}noise')
    plt.clf()


def plot_best_T_boundaries(classifier, X_train, y_train, test_error,
                           noise_ratio):
    """
    Find the T for which the test error was minimal, and plot the decision
    boundaries of the classifier with the same T and the train data.
    :param noise_ratio: The noise ratio the data was generated with.
    :param classifier: The classifier to plot for.
    :param X_train: The training data.
    :param y_train: The training labels.
    :param test_error: The test error rates.
    """
    T_hat = np.argmin(test_error)
    ex4_tools.decision_boundaries(classifier, X_train, y_train, T_hat)
    plt.savefig(f'Plots/best_t_{int(noise_ratio*100)}noise')
    plt.clf()


def plot_weighted_boundaries(classifier, X_train, y_train, noise_ratio):
    """
    Plot the decision boundaries of the last Adaboost iteration, with
    consideration of the weights calculated by the classifier.
    :param noise_ratio: The noise ratio the data was generated with.
    :param classifier: The Adaboost classifier.
    :param X_train: The train data.
    :param y_train: The train labels.
    """
    D = classifier.Dt / np.max(classifier.Dt) * 10
    ex4_tools.decision_boundaries(classifier, X_train, y_train, T, D)
    plt.savefig(f'Plots/weighted_boundaries_{int(noise_ratio*100)}noise')
    plt.clf()


if __name__ == '__main__':
    T = 500
    num_of_train_samples = 5000
    num_of_test_samples = 200
    decision_boundaries_T = [5, 10, 50, 100, 200, 500]
    train_and_plot(0)
    train_and_plot(0.01)
    train_and_plot(0.4)
