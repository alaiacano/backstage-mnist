import datetime
import sys
import mlflow
import mlflow.keras

import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from backstage_utils import (
    extract_experiment_name,
    BASE_DIR,
    EVALUATION_SET_TAG,
    NOTE_TAG,
    TRACKING_URI,
)

# Code example taken from https://towardsdatascience.com/xgboost-python-example-42777d01001e

if __name__ == "__main__":
    experiment_name = extract_experiment_name(f"{BASE_DIR}/component-info-xgboost.yaml")
    if not experiment_name:
        print("No MLFlow Experiment name found. Exiting.")
        sys.exit(1)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name=f"xgboost example on {str(datetime.datetime.now())}"):
        mlflow.xgboost.autolog()

        print("Loading dataset")
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        y = pd.Series(boston.target)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        mlflow.set_tag(EVALUATION_SET_TAG, "25 percent Split")

        print("building regressor")
        regressor = xgb.XGBRegressor(
            n_estimators=100,
            reg_lambda=1,
            gamma=0,
            max_depth=3
        )

        print("fitting model")
        regressor.fit(X_train, y_train)
        
        print("predicting on eval set")
        y_pred = regressor.predict(X_test)

        # Log the mean squared error
        print("calculating MSE")
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)

        # Save the model artifact
        print("Saving model")
        regressor.save_model(f"{BASE_DIR}/outputs-xgb/model.json")
        mlflow.log_artifacts(f"{BASE_DIR}/outputs-xgb")
    print("done.")

