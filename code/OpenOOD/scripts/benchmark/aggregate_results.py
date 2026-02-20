#!/usr/bin/env python
"""
Aggregate OpenOOD benchmark results into summary tables.
Run this after all benchmarks complete.
"""

import os
import pandas as pd
from glob import glob
import argparse


def find_result_csvs(results_dir, dataset):
    """Find all result CSV files for a dataset."""
    patterns = [
        f"{results_dir}/{dataset}*/ood/*.csv",
        f"{results_dir}/{dataset}*/s*/ood/*.csv",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob(pattern))
    return files


def parse_csv(csv_path):
    """Parse a result CSV file."""
    try:
        df = pd.read_csv(csv_path, index_col=0)
        return df
    except Exception as e:
        print(f"Warning: Could not parse {csv_path}: {e}")
        return None


def aggregate_dataset_results(results_dir, dataset):
    """Aggregate results for a single dataset."""
    csv_files = find_result_csvs(results_dir, dataset)

    if not csv_files:
        print(f"No results found for {dataset}")
        return None

    results = {}
    for csv_path in csv_files:
        method = os.path.basename(csv_path).replace('.csv', '')
        df = parse_csv(csv_path)
        if df is not None:
            results[method] = df

    return results


def create_summary_table(results, metric='AUROC'):
    """Create summary table for a specific metric."""
    rows = []

    for method, df in results.items():
        if df is None:
            continue

        row = {'Method': method}
        for idx in df.index:
            if metric in df.columns:
                row[idx] = df.loc[idx, metric]
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.set_index('Method')
    return summary


def main():
    parser = argparse.ArgumentParser(description='Aggregate OpenOOD results')
    parser.add_argument('--results-dir', default='./results',
                        help='Directory containing results')
    parser.add_argument('--output-dir', default='./benchmark_results',
                        help='Directory to save aggregated results')
    parser.add_argument('--datasets', nargs='+',
                        default=['cifar10', 'cifar100', 'imagenet200', 'imagenet'],
                        help='Datasets to aggregate')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in args.datasets:
        print(f"\n=== Aggregating {dataset} results ===")

        results = aggregate_dataset_results(args.results_dir, dataset)

        if results:
            # Create summary tables
            for metric in ['AUROC', 'FPR@95', 'AUPR_IN', 'AUPR_OUT']:
                summary = create_summary_table(results, metric)
                if not summary.empty:
                    output_path = os.path.join(
                        args.output_dir,
                        f'{dataset}_{metric.replace("@", "_at_")}_summary.csv'
                    )
                    summary.to_csv(output_path)
                    print(f"Saved: {output_path}")

                    # Print summary
                    print(f"\n{dataset} - {metric}:")
                    print(summary.to_string())

    print(f"\n=== Aggregation Complete ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
