# California Housing Price Prediction

This repository contains a simple machine learning project that uses the California Housing dataset to train a linear regression model and make predictions on housing prices. The goal of this project is to demonstrate the use of [MLflow](https://mlflow.org/) for model versioning and tracking.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- MLflow
- scikit-learn
- pandas

### Installing

Clone the repository and install the required packages:
```
git clone https://github.com/umairahmad-ua/california-housing-price-prediction.git
cd california-housing-price-prediction
pip install -r requirements.txt
```


### Running the code

The main script, `train.py`, is responsible for training the model and logging the results to an MLflow run. You can execute the script with:
```
mlflow run . -P alpha=0.5 -P l1_ratio=0.5
```

This will run the script with the parameters `alpha=0.5` and `l1_ratio=0.5`. You can change these values as you wish. The script will log the following information to an MLflow run:

- Model parameters
- Metrics (R2 score)
- Artifacts (trained model)

To see the logged information, open the MLflow UI by running:
```
mlflow ui
```

And navigate to the `http://localhost:5000` in your browser.

## Versioning

MLflow allows you to version your models, which is useful when you want to compare different versions of a model or roll back to a previous version. To create a new version of the model, you can use the `mlflow models create` command. For example, to create a new version of the model with the run ID `1` and the version `2`, you can use the following command:
```
mlflow models create -m runs:/1 -v 2
```

You can then use the `mlflow models serve` command to deploy the model to a production environment.

## Conclusion

This project demonstrates the use of MLflow for model versioning and tracking. The California Housing dataset was used to train a linear regression model, and the results were logged to an MLflow run. You can use the MLflow UI to view the results, create new versions of the model, and deploy the model to a production environment.

Please note that you need to replace the [username] in the git clone command with your actual username.
