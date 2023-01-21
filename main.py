from mlflow import log_metric, log_param, log_artifacts
from sklearn.model_selection import train_test_split
from sklearn import datasets
from training import training, testing, dump_model

def get_data():
    # load the boston dataset
    boston = datasets.fetch_california_housing(return_X_y=False)

    # defining feature matrix(X) and response vector(y)
    X = boston.data
    y = boston.target
    return X, y

if __name__=="__main__":
     # downloading data
    X, y = get_data()
    # splitting X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                        random_state=1)
    model = training(X_train, y_train)
    coef_, intercept_, score = testing(model, X_test, y_test)
    dump_model(model)


