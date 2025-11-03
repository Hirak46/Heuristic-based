"""trial: A Flower / PyTorch app with attack logging and rich metrics."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from trial.task import Net, get_weights

round_counter = 0
BASE_LOG_DIR = Path("logs")
BASE_LOG_DIR.mkdir(exist_ok=True)
RUN_LOG_DIR = None  # Will be set in server_fn
METRICS_CSV = None  # Depends on run
PARTICIPATION_CSV = None


def _init_csv():
    if METRICS_CSV and not METRICS_CSV.exists():
        with METRICS_CSV.open('w', newline='') as f:
            csv.writer(f).writerow(['round','accuracy','precision','recall','f1'])
    if PARTICIPATION_CSV and not PARTICIPATION_CSV.exists():
        with PARTICIPATION_CSV.open('w', newline='') as f:
            csv.writer(f).writerow(['round','sampled_clients','malicious_sampled'])


def get_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    global round_counter
    round_counter += 1
    total_examples = sum(num for num, _ in metrics)
    def wavg(key: str):
        return sum(m[key]*n for n, m in metrics)/total_examples if total_examples>0 else 0.0
    accuracy = wavg('accuracy')
    precision = wavg('precision')
    recall = wavg('recall')
    f1 = wavg('f1')
    # Aggregate confusion matrix counts if present
    cm_accum = np.zeros((10,10), dtype=np.int64)
    for n, m in metrics:
        # If this client provided confusion entries
        cell_keys = [k for k in m.keys() if k.startswith('cm_')]
        if cell_keys:
            for k in cell_keys:
                try:
                    _, i, j = k.split('_')
                    cm_accum[int(i), int(j)] += int(m[k])
                except Exception:
                    pass
    _init_csv()
    if METRICS_CSV:
        with METRICS_CSV.open('a', newline='') as f:
            csv.writer(f).writerow([round_counter, accuracy, precision, recall, f1])
    # Write confusion matrix for this round
    if RUN_LOG_DIR and cm_accum.sum() > 0:
        cm_path = RUN_LOG_DIR / f"confusion_round_{round_counter:03d}.csv"
        with cm_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'class_{j}' for j in range(10)])
            for i in range(10):
                writer.writerow(list(cm_accum[i]))
    return {'accuracy': accuracy,'precision': precision,'recall': recall,'f1': f1}


def server_fn(context: Context):
    # Read from config
    global RUN_LOG_DIR, METRICS_CSV, PARTICIPATION_CSV
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    malicious_fraction = context.run_config.get("malicious-fraction", 0.0)
    run_name = context.run_config.get("run-name", f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    RUN_LOG_DIR = BASE_LOG_DIR / run_name
    RUN_LOG_DIR.mkdir(exist_ok=True, parents=True)
    METRICS_CSV = RUN_LOG_DIR / "global_metrics.csv"
    PARTICIPATION_CSV = RUN_LOG_DIR / "participation.csv"

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    def fit_config_fn(server_round: int):
        return {"malicious-fraction": malicious_fraction,"round": server_round}

    def eval_config_fn(server_round: int):
        return {"malicious-fraction": malicious_fraction,"round": server_round}
    
    class TrackingFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            # Each result: (client_proxy, FitRes)
            malicious_count = 0
            for client_proxy, fit_res in results:
                try:
                    if fit_res.metrics.get('malicious', 0) == 1:
                        malicious_count += 1
                except Exception:
                    pass
            sampled_clients = len(results)
            _init_csv()
            if PARTICIPATION_CSV:
                with PARTICIPATION_CSV.open('a', newline='') as f:
                    csv.writer(f).writerow([server_round, sampled_clients, malicious_count])
            return super().aggregate_fit(server_round, results, failures)

    strategy = TrackingFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=get_weighted_average,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=eval_config_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
