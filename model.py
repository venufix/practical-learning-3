from sklearn.linear_model import LinearRegression


class Model:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


