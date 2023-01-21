from mlflow import log_metric, log_param, log_artifacts

if __name__=="__main__":
     # Log a parameter (key-value pair)
    log_param("param1", 2)

    # Log a metric; metrics can be updated throughout the run
    log_metric('Accuracy', 0.98)
