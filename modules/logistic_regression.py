class LogisticRegression:
    def __init__(
        self, rate=0.01, epochs=200, regularization=0, normalize=True, poly_power=2
    ):
        """
        Basic constructor for logistic regression.

        Parameters
        ----------
        rate : float (default 0.01)
            Learning rate.

        epochs : int (default 200)
            Quantity of gradient steps (iterations).

        regulatization : float (default 0)
            Avoid overfitting during learning.

        normalize : bool (default True)
            Normalization data to boost convegence.

        poly_power : int (default 2)
            Max polynomial power of features.

        """

        self.rate = rate
        self.regularization = regularization
        self.epochs = epochs
        self.Xtrain = None
        self.normalize = normalize
        self.ytrain = None
        self.Xtest = None
        self.ytest = None
        self.theta = None
        self.poly_power = poly_power
        self.history = []

    @staticmethod
    def hypothesis(X, theta):
        """
        Compute hypothesis using sigmoid function.

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        theta : numpy.ndarray
            Weights for logistic regression.

        Returns
        -------
        numpy.ndarray
            Hypothesis or predicted vector of values.

        """

        h = 1 / (1 + np.e ** (-np.dot(X, theta)))
        h[np.where(h == 0)] = 10 ** -9
        return h

    def cost_function(self):
        """Compute loss using cost function."""
        h = self.hypothesis(self.Xtrain, self.theta)
        r = self.regularization / (2 * len(self.Xtrain)) * sum(self.theta ** 2)
        cost = (
            -1
            / len(self.Xtrain)
            * sum(
                np.dot(self.ytrain.T, np.log(h))
                + np.dot((1 - self.ytrain).T, np.log(1 - h))
            )
            + r
        )
        self.history.append(*cost)
        return cost

    def gradient_step(self):
        """Implement one gradient step to the local(global) minimum."""
        self.theta = (
            self.theta
            - self.rate
            / len(self.Xtrain)
            * sum(
                (self.hypothesis(self.Xtrain, self.theta) - self.ytrain) * self.Xtrain
            ).reshape(self.theta.shape)
            + self.regularization / len(self.Xtrain) * sum(self.theta)
        )

    def gradient_descent(self):
        """Compute gradient descent."""
        for epoch in range(self.epochs):
            self.gradient_step()
            print("Epoch: {} Cost: {}".format(epoch + 1, *self.cost_function()))

    @staticmethod
    def normalization(Xtrain: np.ndarray) -> np.ndarray:
        # normalize data.
        return (Xtrain - Xtrain.mean(axis=0)) / (
            Xtrain.max(axis=0) - Xtrain.min(axis=0)
        )

    def _preparation_data(self, Xtrain, ytrain):
        """
        Preprocess data and randomize weights.

        Parameters
        ----------
        Xtrain : numpy.ndarray, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        ytrain : numpy.ndarray, shape (n_samples, 1)
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """

        if self.normalize:
            Xtrain = self.normalization(Xtrain)

        for i in range(2, self.poly_power + 1):
            new_feature = Xtrain[:, :2] ** i
            Xtrain = np.concatenate([Xtrain, new_feature], axis=1)

        self.Xtrain = np.concatenate([np.ones((Xtrain.shape[0], 1)), Xtrain], axis=1)
        self.ytrain = ytrain
        self.theta = np.random.randn(self.Xtrain.shape[1], 1)

    def fit(self, Xtrain, ytrain):
        """
        Prepare data and train model.

        Parameters
        ----------
        Xtrain : numpy.ndarray, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        ytrain : numpy.ndarray, shape (n_samples, 1)
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        self._preparation_data(Xtrain, ytrain)
        self.gradient_descent()

    def show_history(self):
        """Plot the history of the cost function during all time of training."""

        plt.plot(np.arange(0, self.epochs, 1), self.history, c="r")
        plt.grid(1)
        plt.legend(["Loss"])

    def show_plot(self):
        """Plot the model and data points."""

        pos = np.where(self.ytrain == 1)[0]
        neg = np.where(self.ytrain == 0)[0]

        plt.scatter(
            self.Xtrain[:, 1][neg], self.Xtrain[:, 2][neg], c="g", marker="+", s=60
        )
        plt.scatter(
            self.Xtrain[:, 1][pos], self.Xtrain[:, 2][pos], c="b", marker="*", s=60
        )
        plt.grid(1)
        new_arange = np.arange(
            min(self.Xtrain[:, 1]), max(self.Xtrain[:, 1]) + 0.2, 0.2
        )
        plt.plot(
            new_arange,
            -self.theta[0] / self.theta[2] - self.theta[1] / self.theta[2] * new_arange,
            c="r",
        )
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend(["Yes", "No", "Line"])
        plt.title("Plot data")
