#!/usr/bin/python3
'''
Entrypoint for the compute package.
'''
import os
import sys
import argparse
import json
import yaml

from model import create_submission, train_model
from preprocess import (filter_dataframe, group_and_aggregate, analyze_column,
                        handle_missing_values, combine_dataframes)


def run_dataset_action(cmd: str, filepath: str):
    """
    Runs generic dataset preprocessing action.

    Parameters
    ----------
    cmd: `str`
    The action name.

    filepath: `str`
    The dataset filepath in the DFS.
    """
    return {
    "filter_dataframe": filter_dataframe,
    "group_and_aggregate": group_and_aggregate,
    "analyze_column": analyze_column,
    "handle_missing_values": handle_missing_values,
    "combine_dataframes": combine_dataframes,
    }[cmd](filepath)


def print_output(data: dict):
    """
    Creates a marked section in the standard output
    of the container in order for Brane to isolate the result.

    Parameters
    ----------
    data: `dict`
    Any valid Python dictionary that is YAML serializable.
    """
    print("--> START CAPTURE")
    print(yaml.dump(data))
    print("--> END CAPTURE")


def main():
    parser = argparse.ArgumentParser(description="Dataset Processing Script")

    parser.add_argument("cmd", type=str, choices=[
        "filter_dataframe",
        "group_and_aggregate",
        "analyze_column",
        "handle_missing_values",
        "combine_dataframes"
    ], help="Command to execute")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")

    # Optional arguments for specific commands
    parser.add_argument("--column", type=str, help="Column name for filtering, analyzing, or missing value handling")
    parser.add_argument("--condition", type=str, help="Condition for filtering (should be a lambda expression as string)")
    parser.add_argument("--group_by_column", type=str, help="Column to group by for aggregation")
    parser.add_argument("--agg_column", type=str, help="Column to aggregate")
    parser.add_argument("--agg_func", type=str, choices=["sum", "mean", "count", "max", "min"], help="Aggregation function")
    parser.add_argument("--method", type=str, choices=["drop", "fill"], help="Method to handle missing values")
    parser.add_argument("--fill_value", type=str, help="Value to fill missing data")
    parser.add_argument("--dataset_path_2", type=str, help="Second dataset path for merging")
    parser.add_argument("--on_column", type=str, help="Column to merge on")
    parser.add_argument("--how", type=str, choices=["inner", "outer", "left", "right"], default="inner", help="Merge type")

    args = parser.parse_args()


    # Execute the appropriate function
    if args.cmd == "filter_dataframe":
        condition_func = eval(args.condition) if args.condition else lambda x: True
        result_path = filter_dataframe(args.dataset_path, args.column, condition_func)
    
    elif args.cmd == "group_and_aggregate":
        result_path = group_and_aggregate(args.dataset_path, args.group_by_column, args.agg_column, args.agg_func)
    
    elif args.cmd == "analyze_column":
        result_path = analyze_column(args.dataset_path, args.column)
    
    elif args.cmd == "handle_missing_values":
        fill_value = eval(args.fill_value) if args.fill_value else None
        result_path = handle_missing_values(args.dataset_path, args.method, fill_value)
    
    elif args.cmd == "combine_dataframes":
        result_path = combine_dataframes(args.dataset_path, args.dataset_path_2, args.on_column, args.how)

    print(f"Processed dataset saved at: {result_path}")

if __name__ == '__main__':
    main()
