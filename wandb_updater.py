import wandb
import numpy as np
import statistics

from collections import defaultdict
from config import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME

ENTITY = WANDB_ENTITY_NAME
PROJECT = WANDB_PROJECT_NAME
METRIC_NAME_LIST = ["test_precision", "test_recall", "test_f1", "test_loss"]

api = wandb.Api(api_key="f904ed1462c53edef7fef2f82b6c04e99ea34339")

runs = api.runs(f"{ENTITY}/{PROJECT}")

experiment_ids = set()


# Primo passaggio: raccogliere tutti gli experiment_id
for run in runs:
    experiment_id = run.config.get("experiment_id")
    if experiment_id:
        experiment_ids.add(experiment_id)

# Secondo passaggio: raccogliere metriche per experiment_id
for METRIC_NAME in METRIC_NAME_LIST:
    metrics = defaultdict(list)
    for run in runs:
        experiment_id = run.config.get("experiment_id")
        if experiment_id:
            try:
                metrics[experiment_id].append(run.summary[METRIC_NAME])
            except KeyError:
                print(f"Run {run.id} does not have the metric '{METRIC_NAME}' in its summary. Skipping...")
                break

    # Calcolo delle statistiche
    maximums = {}
    minimums = {}
    std_devs = {}

    for exp_id in metrics.keys():
        try:
            maximums[exp_id] = np.max(metrics[exp_id])
            minimums[exp_id] = np.min(metrics[exp_id])
            # std_devs[exp_id] = np.std(metrics[exp_id], ddof=1)  # ddof=1 for sample standard deviation
            std_devs[exp_id] = statistics.stdev(metrics[exp_id])  # Using statistics.stdev for sample standard deviation
            average = np.mean(metrics[exp_id])
        except ValueError:
            print(f"Experiment {exp_id} has no valid metrics. Skipping...")

    # Aggiornamento dei summary delle run
    for run in runs:
        experiment_id = run.config.get("experiment_id")
        if experiment_id:
            try:
                run.summary[f"{METRIC_NAME}_max"] = maximums[experiment_id]
                run.summary[f"{METRIC_NAME}_min"] = minimums[experiment_id]
                run.summary[f"{METRIC_NAME}_std"] = std_devs[experiment_id]
            except KeyError:
                print(f"Experiment {experiment_id} not found in metrics. Skipping run {run.id}...")
            run.summary.update()
