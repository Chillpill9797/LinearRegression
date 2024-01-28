import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("test_scores.csv")
r = LinearRegression()
r.fit(df[['math']], df.cs)

def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_predicted)])
        md = (2/n)*sum(x*(y_predicted - y))
        bd = (2/n)*sum(y_predicted - y)
        m_curr = m_curr - learning_rate*md
        b_curr = b_curr - learning_rate*bd
        print(f"m {m_curr}, b {b_curr}, cost {cost}, iteration {i}")

x = np.array(df.math)
y = np.array(df.cs)
gradient_descent(x, y)