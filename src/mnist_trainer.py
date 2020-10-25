import sys
import tensorflow as tf

# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import mlflow
import mlflow.keras

from backstage_utils import (
    extract_experiment_name,
    EVALUATION_SET_TAG,
    NOTE_TAG,
    TRACKING_URI,
)

BASE_DIR = "/Users/adam/github/alaiacano/backstage-mnist"

if __name__ == "__main__":
    experiment_name = extract_experiment_name(f"{BASE_DIR}/component-info.yaml")
    if not experiment_name:
        print("No MLFlow Experiment name found. Exiting.")
        sys.exit(1)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name=experiment_name)

    # Capture tensorflow metrics
    mlflow.keras.autolog()

    with mlflow.start_run():
        print("STARTING RUN")
        mlflow.set_tags(
            {
                EVALUATION_SET_TAG: "keras-mnist-test-set",
                NOTE_TAG: "using keras autolog instead of tensorflow",
            }
        )

        print("LOADING DATA")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        print("RESHAPING DATA")
        # Reshaping the array to 4-dims so that it can work with the Keras API
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
        # Making sure that the values are float so that we can get decimal points after division
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train /= 255
        x_test /= 255

        print("BUILDING MODEL")
        # Creating a Sequential Model and adding the layers
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=tf.nn.softmax))

        print("COMPILING MODEL")
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("TRAINING MODEL")
        model.fit(x=x_train, y=y_train, epochs=10)

        # Model evaluation metrics (loss, accuracy) are
        # already done via mlflow.keras.autolog()

        print("SAVING AND LOGGING ARTIFACT")
        model.save(f"{BASE_DIR}/outputs")
        mlflow.log_artifacts(f"{BASE_DIR}/outputs")
