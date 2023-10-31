import os

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(data_path, src_path, images_path, train_path, test_path, test_size):
    if os.path.exists(data_path + train_path) and os.path.exists(
        data_path + test_path
    ):  # If the data has already been processed
        train_df = pd.read_csv(data_path + train_path)
        test_df = pd.read_csv(data_path + test_path)
    else:
        # Read the data
        df = pd.read_csv(data_path + src_path)

        # Split the data into training, validation, and test sets
        train_df, test_df = train_test_split(df, test_size=test_size)

        # Prepend the path to the images
        train_df["file"] = images_path + train_df["file"]
        test_df["file"] = images_path + test_df["file"]

        # Save the data
        train_df.to_csv(data_path + train_path, index=False)
        test_df.to_csv(data_path + test_path, index=False)

    return train_df, test_df
