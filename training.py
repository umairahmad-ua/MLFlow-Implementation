from sklearn import linear_model
import os
import pickle


def training(X_train, y_train):
    # create linear regression object
    model = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, positive=False)

    # train the model using the training sets
    model.fit(X_train, y_train)
    return model


def testing(model, X_test, y_test):
    # regression coefficients
    print('Coefficients: ', model.coef_)

    # checking intercept - b
    print('intercept: ', model.intercept_)

    # variance score: 1 means perfect prediction
    score = model.score(X_test, y_test)
    print('Variance score: {}'.format(score))

    return model.coef_, model.intercept_, score


def dump_model(model):
    if not os.path.exists("model"):
        os.makedirs("model")
    with open('model/linear_regression_model.pkl', 'wb') as files:
        pickle.dump(model, files)

