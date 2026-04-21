import json
import numpy
import sklearn
from sklearn import linear_model
import os

def simple_predictor(data):
    ratings = [d['rating'] for d in data]
    lengths = [len(d['review_text']) for d in data]
    
    X = numpy.array([[1, l] for l in lengths])
    y = numpy.array(ratings)

    model = sklearn.linear_model.LinearRegression(fit_intercept=False)
    model.fit(X, y)
    theta = model.coef_  
    y_pred = model.predict(X)
    
    return theta, y, y_pred 


def mean_squared_error(y_actual, y_pred):
    return numpy.mean((y_actual - y_pred) ** 2)


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "data", "fantasy_100.json") 

    f = open(path)
    data = []
    for l in f:
        d = json.loads(l)
        data.append(d)
    f.close()

    theta, y, y_pred = simple_predictor(data)
    print(f"theta0 = {theta[0]:.4f}, theta1 = {theta[1]:.4f}")
    print(f"MSE = {mean_squared_error(y, y_pred):.4f}")


if __name__ == "__main__":
    main()