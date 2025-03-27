import pandas as pd

def filter_dataframe(dataset_path: str, column: str, condition) -> str:
    """
    Filters a DataFrame based on a given condition.

    Parameters
    ----------
    dataset_path: `str`
        Path to the dataset.
    column: `str`
        Column name to filter on.
    condition:
        A lambda function or value to filter the column.

    Returns
    -------
    `str` The path of the filtered dataset.
    """
    df = pd.read_csv(dataset_path)
    df_filtered = df[df[column].apply(condition)]
    new_path = "/result/filtered_dataset.csv"
    df_filtered.to_csv(new_path, index=False)
    return new_path


def group_and_aggregate(dataset_path: str, group_by_column: str, agg_column: str, agg_func: str) -> str:
    """
    Groups a DataFrame by a column and aggregates another column.

    Parameters
    ----------
    dataset_path: `str`
        Path to the dataset.
    group_by_column: `str`
        The column to group by.
    agg_column: `str`
        The column to apply the aggregation function on.
    agg_func: `str`
        The aggregation function ('sum', 'mean', 'count', etc.).

    Returns
    -------
    `str` The path of the grouped and aggregated dataset.
    """
    df = pd.read_csv(dataset_path)
    df_grouped = df.groupby(group_by_column)[agg_column].agg(agg_func).reset_index()
    new_path = "/result/grouped_dataset.csv"
    df_grouped.to_csv(new_path, index=False)
    return new_path


def analyze_column(dataset_path: str, column: str) -> str:
    """
    Analyzes a column by providing summary statistics.

    Parameters
    ----------
    dataset_path: `str`
        Path to the dataset.
    column: `str`
        The column to analyze.

    Returns
    -------
    `str` The path of the analyzed column output.
    """
    df = pd.read_csv(dataset_path)
    summary = df[column].describe().to_dict()
    unique_values = df[column].nunique()

    result = {"summary_statistics": summary, "unique_value_count": unique_values}
    
    # Saving as JSON
    new_path = "/result/column_analysis.json"
    with open(new_path, "w") as f:
        json.dump(result, f)

    return new_path


def handle_missing_values(dataset_path: str, method: str = "drop", fill_value=None) -> str:
    """
    Handles missing values in the dataset.

    Parameters
    ----------
    dataset_path: `str`
        Path to the dataset.
    method: `str`
        Method to handle missing values: "drop" to remove rows, "fill" to replace with `fill_value`.
    fill_value: `any`
        Value to replace NaNs if using the "fill" method.

    Returns
    -------
    `str` The path of the dataset with missing values handled.
    """
    df = pd.read_csv(dataset_path)

    if method == "drop":
        df = df.dropna()
    elif method == "fill" and fill_value is not None:
        df = df.fillna(fill_value)

    new_path = "/result/cleaned_dataset.csv"
    df.to_csv(new_path, index=False)
    return new_path


def combine_dataframes(dataset_path_1: str, dataset_path_2: str, on_column: str, how: str = "inner") -> str:
    """
    Merges two DataFrames on a common column.

    Parameters
    ----------
    dataset_path_1: `str`
        Path to the first dataset.
    dataset_path_2: `str`
        Path to the second dataset.
    on_column: `str`
        The common column to merge on.
    how: `str`
        Merge type: "inner", "outer", "left", "right".

    Returns
    -------
    `str` The path of the merged dataset.
    """
    df1 = pd.read_csv(dataset_path_1)
    df2 = pd.read_csv(dataset_path_2)

    df_merged = df1.merge(df2, on=on_column, how=how)
    
    new_path = "/result/merged_dataset.csv"
    df_merged.to_csv(new_path, index=False)
    return new_path
