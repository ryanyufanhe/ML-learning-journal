from linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def test_logistic_regression():
    data = load_iris()
    x = data["data"]
    y = data["target"]

    pca = PCA(n_components=2)
    pca.fit(x)
    x = pca.transform(x)

    for i in range(len(x)):
        if y[i] == 0:
            plt.scatter(x[i][0], x[i][1], color="red")
        elif y[i] == 1:
            plt.scatter(x[i][0], x[i][1], color="green")
        elif y[i] == 2:
            plt.scatter(x[i][0], x[i][1], color="blue")

    x1_train = x[:100]
    y1_train = y[:100]
    lr_1 = LogisticRegression()
    lr_1.fit(x1_train, y1_train)

    x2_train = x[50:]
    y2_train = y[50:] - 1
    lr_2 = LogisticRegression()
    lr_2.fit(x2_train, y2_train)

    x_test = np.linspace(-3.5, 4, 200)
    y1_test = (-lr_1._intercept[0] - lr_1._coef[0] * x_test) / lr_1._coef[1]
    plt.plot(x_test, y1_test)

    y2_test = (-lr_2._intercept[0] - lr_2._coef[0] * x_test) / lr_2._coef[1]
    plt.plot(x_test, y2_test)

    plt.xlim((-3.5, 4))
    plt.ylim((-1.5, 1.5))
    plt.show()


if __name__ == "__main__":
    test_logistic_regression()
