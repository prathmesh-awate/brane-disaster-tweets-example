import ast
import pickle
import re
from typing import List

import nltk
import pandas as pd
import sklearn
import json

# download preprocessing assets (corpus and word lists)
# NOTE: Due to the new no-internet constraints during runtime, these are actually downloaded beforehand in the `container.yml`
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def clean(dataset_path: str) -> str:
    """
    Applies regex-based text cleaning to the 'text' column
    for every dataset row.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'text' column which contains strings.

    Returns
    -------
    `str` The path for the clean version of the dataset in the DFS.
    """
    def _remove_unused(text: str):
        clean_data = text.lower().strip()
        clean_data = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            " ", clean_data)
        clean_data = re.sub(r"<.*>", "", clean_data)
        clean_data = re.sub(r"@[a-zA-Z0-9_]+", "", clean_data)
        clean_data = clean_data.replace("\n", "")\
            .replace("#", "")
        return clean_data

    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str
    }

    dataset_path = f"{dataset_path}/dataset.csv"
    if "train" in dataset_path:
        dtypes["target"] = int

    new_path = "/result/dataset.csv"
    df = pd.read_csv(dataset_path, index_col="id", dtype=dtypes)
    df["text"] = df["text"].apply(_remove_unused)
    df.to_csv(new_path)
    return new_path

def tokenize(dataset_path: str) -> str:
    """
    Creates an additional columns 'tokens' to the dataset.
    It contains a list of stemmed and lemmatized tokens.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'text' column which contains strings.

    Returns
    -------
    `str` The path for the tokenized version of the dataset in the DFS.
    """
    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str
    }

    dataset_path = f"{dataset_path}/dataset.csv"
    if "train" in dataset_path:
        dtypes["target"] = int

    new_path = "/result/dataset.csv"
    df = pd.read_csv(dataset_path, index_col="id", dtype=dtypes)
    df["text_stemmed"] = df["text"].apply(nltk.stem.PorterStemmer().stem)
    df["text_lemmatized"] = df["text_stemmed"].apply(
        nltk.stem.WordNetLemmatizer().lemmatize)
    df["tokens"] = df["text_lemmatized"].apply(
        nltk.tokenize.RegexpTokenizer(r'\w+').tokenize)
    df.to_csv(new_path)
    return new_path

def remove_stopwords(dataset_path: str) -> str:
    """
    Applies stopwords removal to the 'tokens' colums
    for each dataset row.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'tokens' column which contains
    a list of strings.

    Returns
    -------
    `str` The path for the new version of the dataset in the DFS.
    """
    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str,
        "text_stemmed": str,
        "text_lemmatized": str,
    }

    dataset_path = f"{dataset_path}/dataset.csv"
    if "train" in dataset_path:
        dtypes["target"] = int

    def _rm_stopwords(tokens: List[str]):
        return [w for w in tokens
                if w not in nltk.corpus.stopwords.words('english')]

    new_path = "/result/dataset.csv"
    df = pd.read_csv(
        dataset_path,
        index_col="id",
        dtype=dtypes,
        converters={"tokens": ast.literal_eval})
    df["tokens"] = df["tokens"].apply(_rm_stopwords)
    df.to_csv(new_path)
    return new_path

def create_vectors(
    dataset_path_train: str, dataset_path_test: str,
    vectors_path_train: str, vectors_path_test: str
) -> int:
    """
    Converts the raw dataset tweets to token-count vectors. The
    dictionary used for the frequencies is based on the training data.

    Parameters
    ----------
    dataset_path_train: `str`
    The preprocessed training dataset CSV/TSV path in the distributed
    file system.

    dataset_path_test: `str`
    The preprocessed testing dataset CSV/TSV path in the distributed
    file system.

    vectors_path_train: `str`
    The final location for training vectors in the distributed
    file system where the binary file will be stored.

    vectors_path_test: `str`
    The final location for testing vectors in the distributed
    file system where the binary file will be stored.

    Returns
    -------
    `int` Error code (success = 0, failure >= 1)
    """
    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str,
        "text_stemmed": str,
        "text_lemmatized": str,
    }

    df_train = pd.read_csv(
        dataset_path_train,
        index_col="id",
        dtype={**dtypes, "target": int},
        converters={"tokens": ast.literal_eval})
    df_train["text_preprocessed"] = df_train["tokens"].apply(
        lambda x: " ".join(x))

    df_test = pd.read_csv(
        dataset_path_test,
        index_col="id",
        dtype=dtypes,
        converters={"tokens": ast.literal_eval})
    df_test["text_preprocessed"] = df_test["tokens"].apply(
        lambda x: " ".join(x))

    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    vectors_train = vectorizer.fit_transform(df_train["text_preprocessed"])
    vectors_test = vectorizer.transform(df_test["text_preprocessed"])

    with open(vectors_path_train, "wb") as f:
        pickle.dump(vectors_train, f)
    with open(vectors_path_test, "wb") as f:
        pickle.dump(vectors_test, f)

    return 0

def generate_bigrams(dataset_path: str) -> str:
    """
    Generates bi-gram information for each tweet.

    Parameters
    ----------
    dataset_path: `str`
    The dataset CSV/TSV path in the distributed file system.
    It expects a dataset with a 'text' column which contains strings.

    Returns
    -------
    `str`
    The path for the new version of the dataset (with bigrams) in the DFS.
    """
    dtypes = {
        "id": int,
        "keyword": str,
        "location": str,
        "text": str,
    }

    dataset_path = f"{dataset_path}/dataset.csv"
    if "train" in dataset_path:
        dtypes["target"] = int

    def _make_string_bigrams(tokens: List[str]) -> List[str]:
        bigrams = list(nltk.bigrams(tokens))
        out: List[str] = []
        for b in bigrams:
            out.append(f"{b[0]}_{b[1]}")
        return out

    new_path = "/result/dataset.csv"
    df = pd.read_csv(
        dataset_path,
        index_col="id",
        dtype=dtypes,
        converters={"tokens": ast.literal_eval})

    df["bigrams"] = df["tokens"].apply(_make_string_bigrams)
    df.to_csv(new_path)

    return new_path


def filter_dataframe(dataset_path: str) -> str:
    """
    Filters rows where median_income > 5 in the given dataset.

    Parameters
    ----------
    dataset_path: `str`
        The path to the dataset CSV file.

    Returns
    -------
    `str` The path to the filtered version of the dataset.
    """
    dataset_path = f"{dataset_path}/california_housing_test.csv"
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Apply filtering
    filtered_df = df[df['median_income'] > 5]

    # Save the filtered dataset to a new file
    filtered_path = "/result/filtered_dataset.csv"
    filtered_df.to_csv(filtered_path)
    
    return filtered_path


def group_and_aggregate(dataset_path: str) -> str:
    """
    Groups data by `housing_median_age` and calculates
    the mean `median_house_value` for each group.

    Parameters
    ----------
    dataset_path: `str`
        The path to the dataset CSV file.

    Returns
    -------
    `str` The path to the aggregated dataset.
    """
    
    dataset_path = f"{dataset_path}/california_housing_test.csv"
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Group and aggregate
    aggregated_df = df.groupby('housing_median_age', as_index=False)['median_house_value'].mean()

    # Save the aggregated dataset
    aggregated_path = "/result/aggregated_dataset.csv"
    aggregated_df.to_csv(aggregated_path, index=False)

    return aggregated_path


def analyze_column(dataset_path: str) -> str:
    """
    Provides descriptive statistics for the `total_rooms` column
    and saves them as a CSV file.

    Parameters
    ----------
    dataset_path: `str`
        The path to the dataset CSV file.

    Returns
    -------
    `str` The path to the CSV file containing column analysis.
    """
    dataset_path = f"{dataset_path}/california_housing_test.csv"
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Analyze the column
    analysis = df['total_rooms'].describe()

    # Save the analysis to a CSV file
    analysis_path = "/result/column_analysis.csv"
    analysis.to_csv(analysis_path, header=True)

    return analysis_path


def handle_missing_values(dataset_path: str) -> str:
    """
    Handles missing values in the `total_bedrooms` column by
    replacing them with the column mean.

    Parameters
    ----------
    dataset_path: `str`
        The path to the dataset CSV file.

    Returns
    -------
    `str` The path to the dataset with missing values handled.
    """
    dataset_path = f"{dataset_path}/california_housing_test.csv"
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Handle missing values
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())

    # Save the updated dataset
    updated_path = "/result/handled_missing_values.csv"
    df.to_csv(updated_path, index=False)

    return updated_path


def combine_dataframes(dataset_path: str) -> str:
    """
    Combines the given dataset with a supplementary dataset containing
    region information based on `longitude`.

    Parameters
    ----------
    dataset_path: `str`
        The path to the dataset CSV file.

    Returns
    -------
    `str` The path to the combined dataset.
    """
    dataset_path = f"{dataset_path}/california_housing_test.csv"
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Define a supplementary dataset
    extra_data = {
        'longitude': [-122.05, -118.3, -117.81],
        'region': ['North', 'South', 'South']
    }
    extra_df = pd.DataFrame(extra_data)

    # Combine the dataframes
    combined_df = pd.merge(df, extra_df, on='longitude', how='left')

    # Save the combined dataset
    combined_path = "/result/combined_dataset.csv"
    combined_df.to_csv(combined_path, index=False)

    return combined_path

