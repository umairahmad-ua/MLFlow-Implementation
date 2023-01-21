from mlflow import log_metric, log_param, log_artifacts
from sklearn.model_selection import train_test_split
from sklearn import datasets

def get_data():
    # load the boston dataset
    boston = datasets.fetch_california_housing(return_X_y=False)

    # defining feature matrix(X) and response vector(y)
    X = boston.data
    y = boston.target
    return X, y

if __name__=="__main__":
     # Log a parameter (key-value pair)
    log_param("param1", 2)

    # Log a metric; metrics can be updated throughout the run
    log_metric('Accuracy', 0.98)
