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
            metrics[parent_dir].pop('recall', None)
            metrics[parent_dir].pop('precision', None)
            metrics[parent_dir].pop('accuracy', None)
            best_epoch = ""
            with open(Path(parent_dir) / "metrics_k_fold_val.json") as j:
                val_metrics = json.load(j)
                val_metrics = val_metrics['average']
                for epochs in val_metrics:
                    if val_metrics[epochs]['dsc'] == metrics[parent_dir]['dsc']:
                        best_epoch = epochs
            if best_epoch is not "":
                metrics[parent_dir]["val"] = best_epoch
                with open(Path(parent_dir) / "metrics_k_fold_train.json") as j:
                    train_metrics = json.load(j)
                    metrics[parent_dir]["dsc train"] = train_metrics["average"][best_epoch]["dsc"] - metrics[parent_dir]["dsc"]
                    metrics[parent_dir]["loss train"] = train_metrics["average"][best_epoch]["loss"] - metrics[parent_dir]["loss"]
                    metrics[parent_dir]["iou train"] = train_metrics["average"][best_epoch]["iou"] - metrics[parent_dir]["iou"]
            else:
                metrics[parent_dir]["dsc train"] = 0.0
                metrics[parent_dir]["loss train"] = 0.0
                metrics[parent_dir]["iou train"] = 0.0
                metrics[parent_dir]["val"] = ""
            test_file = Path(parent_dir) / "metrics_test.json"
            metrics[parent_dir]["dsc test"] = 0.0
            metrics[parent_dir]["loss test"] = 0.0
            metrics[parent_dir]["iou test"] = 0.0
            metrics[parent_dir]["test"] = ""
            metrics[parent_dir]["dsc test train"] = 0.0
            metrics[parent_dir]["loss test train"] = 0.0
            metrics[parent_dir]["iou test train"] = 0.0
            if test_file.is_file():
                best_test_epoch = ""
                with open(test_file, 'r') as t:
                    test_metrics = json.load(t)
                    best_dsc = 0.0
                    for epochs in test_metrics:
                        if test_metrics[epochs]["dsc"] > best_dsc:
                            best_dsc = test_metrics[epochs]["dsc"]
                            best_test_epoch = epochs
                    if best_test_epoch is not "":
                        metrics[parent_dir]["dsc test"] = test_metrics[best_test_epoch]["dsc"]
                        metrics[parent_dir]["loss test"] = test_metrics[best_test_epoch]["loss"]
                        metrics[parent_dir]["iou test"] = test_metrics[best_test_epoch]["iou"]
                        metrics[parent_dir]["test"] = best_test_epoch
                        with open(Path(parent_dir) / "metrics_train.json") as k:
                            test_train_metrics = json.load(k)
                            metrics[parent_dir]["dsc test train"] = test_train_metrics[best_test_epoch]["dsc"] - test_metrics[best_test_epoch]["dsc"]
                            metrics[parent_dir]["loss test train"] = test_train_metrics[best_test_epoch]["loss"] - test_metrics[best_test_epoch]["loss"]
                            metrics[parent_dir]["iou test train"] = test_train_metrics[best_test_epoch]["iou"] - test_metrics[best_test_epoch]["iou"]

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

