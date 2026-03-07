import mlflow
import pandas as pd

mlflow.set_experiment("YouTube Tutorial")

with mlflow.start_run(run_name="Logging Demo", run_id="0626c2835f3b409bbf4a36492a064db9"):
    mlflow.log_param('learning_rate', 0.03)
    mlflow.log_param('epoch', 100)

    parameters = {
        'learning_rate1': 0.04,
        'epoch1': 200
    }

    mlflow.log_params(parameters)

    mlflow.log_metric('accuracy', 90)

    metrics = {
        'accuracy1': 80
    }

    mlflow.log_metrics(metrics)

    mlflow.log_table(pd.DataFrame({'name': ['John', 'Jane'], 'age': [25, 30]}), 'demo_df.json')