import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math


def predict_using_sklean():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_


def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08

    cost_previous = 0
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * np.sum(np.square(y - y_predicted))
        md = (2/n)*sum(x*(y_predicted - y))
        bd = (2/n)*sum(y_predicted - y)
        m_curr = m_curr - learning_rate*md
        b_curr = b_curr - learning_rate*bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print(f"m {m_curr}, b {b_curr}, cost {cost}, iteration {i}")
    return m_curr, b_curr


if __name__ == "__main__":
    df = pd.read_csv("test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)
    m, b = gradient_descent(x, y)
    print(f"Using gradient descent: Coef {m} Intercept {b}.")

    m_sklearn, b_sklearn = predict_using_sklean()
    print(f"Using sklearn: Coef {m_sklearn} Intercept {b_sklearn}")
