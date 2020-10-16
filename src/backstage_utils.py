from typing import Optional
import yaml

TRACKING_URI = "file:///Users/adam/github/scratch/mlflow-demo/mlruns"
EVALUATION_SET_TAG: str = "mlflow.backstage.evaluation_set"
NOTE_TAG: str = "mlflow.note.content"


def extract_experiment_id(yaml_path: str) -> Optional[str]:
    """
    Reads the supplied backstage yaml configuration file and returns the MLFlow Experiment that it is configured
    to use.
    """
    try:
        with open(yaml_path, "r") as fin:
            parsed = yaml.safe_load(fin.read())
            return parsed["metadata"]["annotations"]["mlflow.org/experiments"]
    except:
        return None
