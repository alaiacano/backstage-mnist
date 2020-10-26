from typing import Optional
import yaml
import mlflow
from mlflow.entities import Experiment

BASE_DIR = "/Users/adam/github/alaiacano/backstage-mnist"
TRACKING_URI = "file:///Users/adam/github/scratch/mlflow-demo/mlruns"
EVALUATION_SET_TAG: str = "mlflow.backstage.evaluation_set"
NOTE_TAG: str = "mlflow.note.content"


def extract_experiment_name(yaml_path: str) -> Optional[str]:
    """
    Reads the supplied backstage yaml configuration file and returns the MLFlow Experiment that it is configured
    to use.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    with open(yaml_path, "r") as fin:
        parsed = yaml.safe_load(fin.read())
        experiment_id = parsed["metadata"]["annotations"]["mlflow.org/experiment"]
        print(experiment_id)
        experiment: Experiment = mlflow.get_experiment(experiment_id)
        return experiment.name

