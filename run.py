import numpy as np
import pandas as pd

class Regressor:
    '''
    Logisitc Regression
    '''

    def __init__(self):
        self.coef_ = None
        self.p_ = None

    def fit(self, X, y):

        # OLS fitting
        beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
        beta = np.concatenate(([0], beta[1:]))
        self.coef_ = beta
        return

    def predict_prob(self, X):
        self.p_ = 1.0/(1.0 + np.e**np.dot(-self.coef_, X.T))
        return self.p_
    
    def predict(self, X):
        self.predict_prob(X)
        return np.where(self.p_ > 0.45, 1, 0)

def preprocess(data_path, concat=True):

    data = pd.read_csv(data_path)

    def preprocess_non_str(d):
        x_d = d.fillna(d.mean())
        x_d = np.array(x_d)
        return x_d/np.max(x_d)

    weighting_factor = [1, 10.0, 1.0, 1, 1, 1]

    # preprocess rule:
    x_sex = data["Sex"] 
    x_sex = x_sex.replace(["male", "female"], [0, 1])
    x_sex *= weighting_factor[0]

    x_class = preprocess_non_str(data["Pclass"])
    x_age = preprocess_non_str(data["Age"])
    x_sibsp = preprocess_non_str(data["SibSp"])
    x_parch = preprocess_non_str(data["Parch"])
    x_fare = preprocess_non_str(data["Fare"])

    x_class *= weighting_factor[1]
    x_age *= weighting_factor[2]
    x_sibsp *= weighting_factor[3]
    x_parch *= weighting_factor[4]
    x_fare *= weighting_factor[5]
    
    X = np.array([x_sex, x_class, x_age, x_sibsp, x_parch, x_fare]).T
    if concat:
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

    y = data.get("Survived", None)
    y = np.array(y) if y is not None else None
    return y, X

def write_submission(predict):

    data = pd.read_csv("data/gender_submission.csv")
    answer = data["Survived"]
    score = np.sum(answer==predict)/predict.shape[0]
    print(score)

    data["Survived"] = predict
    data.to_csv("submission.csv", index=False)
    return 

def regression():

    reg = Regressor()

    y, train_data = preprocess("data/train.csv")
    _, test_data = preprocess("data/test.csv")

    reg.fit(train_data, y)
    k = reg.predict(test_data)
    print(reg.coef_)
    write_submission(k)
    return 


if __name__ == "__main__":
    regression()
