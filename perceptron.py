import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class Perceptron:
    def __init__(self, alpha, epochs):
        self.learning_rate = alpha
        self.n_iterations = epochs
        self.weights = None
        self.bias = None

    # Step function.
    def activation_function(self, input):
        return np.where(input > 0, 1, 0)

    def fit(self, features, targets):
        # Initialize weights and bias with random values uniformally distributed.
        n_features = features.shape[1]
        self.weights = np.random.uniform(size=n_features, low=-0.5, high=0.5)
        self.bias = np.random.uniform(low=-0.5, high=0.5)

        # for each epoch, iterate over all samples.
        for _ in range(self.n_iterations):
            for X_i, X in enumerate(features):
                # compute the input of the activation function that will be used to calculate the error.
                y_predicted = self.activation_function(np.dot(X, self.weights) + self.bias)
                # update weights and bias: w_ij(next) = w_ij + alpha * (y_i — y_hat) * x_i + bias
                self._update_weights(X, targets[X_i], y_predicted)

    def _update_weights(self, X, y, y_predicted):
        # compute the error and update the weights and bias.
        error = y - y_predicted
        weight_adj = self.learning_rate * error
        self.weights = self.weights + weight_adj * X
        self.bias = self.bias + weight_adj

    def predict(self, X):
        y_predicted = self.activation_function(np.dot(X, self.weights) + self.bias) # type: ignore
        return y_predicted

def run_perceptron_example():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=3, n_redundant=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    perceptron = Perceptron(alpha=0.01, epochs=1000)
    perceptron.fit(X_train, y_train)

    y_pred = perceptron.predict(X_test)

    sklearn_perceptron = SklearnPerceptron()
    sklearn_perceptron.fit(X_train, y_train)

    y_pred_sklearn = sklearn_perceptron.predict(X_test)

    accuracy_custom = accuracy_score(y_test, y_pred)
    mse_custom = mean_squared_error(y_test, y_pred)

    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

    print(f"Custom Perceptron Accuracy: {accuracy_custom}, MSE: {mse_custom}")
    print(f"Sklearn Perceptron Accuracy: {accuracy_sklearn}, MSE: {mse_sklearn}")

def train_perceptron_on_digits():
    # Load the digits dataset
    X, y = load_digits(return_X_y=True)
    #X, y = digits.data, digits.target # type: ignore

    # Filter out all digits except 0 and 1
    X = X[(y == 0) | (y == 1)]
    y = y[(y == 0) | (y == 1)]

    # Split the data into training, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, train_size=0.6, random_state=42)

    # Initialize and train the perceptron
    perceptron = Perceptron(alpha=0.01, epochs=1000)
    perceptron.fit(X_train, y_train)

    # Use the validation set to choose the model
    y_val_pred = perceptron.predict(X_val)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred)}")

    # Report the selected model’s accuracy using the testing set
    y_test_pred = perceptron.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")


def main():
    #run_perceptron_example()
    train_perceptron_on_digits()

if __name__ == "__main__":
    main()
