"""Aggregates results from the metrics_eval_best_weights.json in a parent folder"""

import argparse
import json

from pathlib import Path

from tabulate import tabulate


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments',
                    help='Directory containing results of experiments')


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.
    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics_dev.json`
    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = Path(parent_dir) / ('metrics_k_fold_val_average_best.json')
    if metrics_file.is_file():
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    # Check every subdirectory of parent_dir
    for subdir in Path(parent_dir).iterdir():
        if not subdir.is_dir():
            continue
        else:
            aggregate_metrics(subdir, metrics)


def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')

    return res


def main(parent_dir):
    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(parent_dir, metrics)
    table = metrics_to_table(metrics)

    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = Path(parent_dir) / "results.md"
    with open(save_file, 'w') as f:
        f.write(table)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.parent_dir)

