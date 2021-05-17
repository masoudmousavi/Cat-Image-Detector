import numpy as np
import h5py
# import matplotlib.pyplot as plt
import json


class LogistincRegressionModel(object):
    def __init__(self):
        self.X, self.Y, self.training_set_size, self.features_count = self._obtain_training_data()
        self.weights, self.bias = self._create_random_params()
        self.Y_hat = self._initialzie_y_hat()
        self.testX, self.testY, self.test_set_size = self._obtain_test_data()


    def _obtain_training_data(self):
        h_file = h5py.File('train_catvnoncat.h5')
        temp_X = h_file.get('train_set_x')[:] / 255
        temp_Y = h_file.get('train_set_y')[:]
        m = temp_X.shape[0]
        X = temp_X.reshape((m, -1)).T
        Y = temp_Y.reshape((m, -1)).T
        m = X.shape[1]
        n_x = X.shape[0]
        return X,\
               Y,\
               m,\
               n_x


    def _obtain_test_data(self):
        h_file = h5py.File('test_catvnoncat.h5')
        temp_X = h_file.get('test_set_x')[:] / 255
        temp_Y = h_file.get('test_set_y')[:]
        m = temp_X.shape[0]
        X = temp_X.reshape((m, -1)).T
        Y = temp_Y.reshape((m, -1)).T
        m_test = X.shape[1]
        return X, \
               Y, \
               m_test


    def _create_random_params(self):
        epsilon = 0.01
        return\
            np.random.randn(self.features_count, 1) * epsilon,\
            0.


    def _initialzie_y_hat(self):
        return \
            np.ones((1, self.training_set_size)) * 0.5


    def cost_function(self):
        cost = np.sum(
                self.Y * np.log(self.Y_hat) + \
                (1 - self.Y) * np.log(1 - self.Y_hat)
            )

        return -cost / self.training_set_size


    def forward_propagation(self, X):
        Z = np.matmul(self.weights.T, X) + self.bias
        sigmoid = np.vectorize(
            lambda z: 1 / (1 + np.exp(-z))
        )
        self.Y_hat = sigmoid(Z)


    def backpropagaion(self):
        loss = self.Y_hat - self.Y
        dweights = np.matmul(self.X, loss.T) / self.training_set_size
        dbias = np.sum(loss) / self.training_set_size
        return dweights,\
               dbias


    def gradient_descent(self):
        # Fully optimizing the parameters are avoided to prevent overfitting
        alpha = 0.002
        for epoch in range(2000):
            self.forward_propagation(self.X)
            dweights, dbias = self.backpropagaion()
            self.weights -= dweights * alpha
            self.bias -= dbias * alpha
            cost = self.cost_function()
            if epoch % 100 == 0:
                print(f'Epoch {epoch+1}, Cost {cost}')
        prob = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        self.Y_hat = prob(self.Y_hat)
        count = 0
        for i in (self.Y - self.Y_hat)[0]:
            if i == 0:
                count += 1

        print(f'Training set accuracy: {np.round(count / self.training_set_size * 100, 2)} %')


    def train(self):
        self.gradient_descent()
        with open('optimized-params.txt', 'w') as fp:

            json.dump({
                'weights': list(np.ravel(self.weights)),
                'bias': list(np.ravel(self.bias))
            }, fp)


    def predict(self):
        with open('optimized-params.txt') as fp:
            data = json.load(fp)
            self.weights = np.array(data['weights']).reshape((self.features_count, 1))
            self.bias = data['bias']
        self.forward_propagation(self.testX)
        prob = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        self.Y_hat = prob(self.Y_hat)
        count = 0
        for i in (self.testY - self.Y_hat)[0]:
            if i == 0:
                count += 1
        print(f'Test set accuracy: {count/self.test_set_size * 100} %')


def main():
    model = LogistincRegressionModel()
    model.train()
    model.predict()


if __name__ == '__main__':
    main()