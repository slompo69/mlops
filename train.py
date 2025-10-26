import os
import random
import mlflow
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# import requests
from urllib.error import HTTPError


def reset_seeds() -> None:
    """
    Resets the seeds to ensure reproducibility of results.
    """
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)


def read_data(use_cache=True):
    """
    Read data with caching to avoid rate limiting
    """
    cache_file = 'fetal_health_cache.csv'

    # Try to use cached data
    if use_cache and os.path.exists(cache_file):
        print("Using cached data...")
        data = pd.read_csv(cache_file)
    else:
        try:
            print("Downloading data from GitHub...")
            domain = 'raw.githubusercontent.com'
            folder = '/renansantosmendes/lectures-cdas-2023/master/'
            file_name = 'fetal_health_reduced.csv'
            url = f'https://{domain}{folder}{file_name}'

            data = pd.read_csv(url)

            # Save to cache for next time
            data.to_csv(cache_file, index=False)
            print("Data cached successfully")

        except HTTPError as e:
            if e.code == 429:
                print("Rate limit exceeded. Using cached data if available...")
                if os.path.exists(cache_file):
                    data = pd.read_csv(cache_file)
                else:
                    # Create sample data as fallback
                    print("No cache available. Creating sample data...")
                    data = create_sample_data()
            else:
                raise

    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    X = data.drop(["fetal_health"], axis=1)
    y = data["fetal_health"]

    return X, y


def create_sample_data():
    """
    Create sample data when download fails and no cache exists
    """
    print("Creating sample data for testing...")
    # Create realistic sample data based on the original dataset structure
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame({
        'baseline value': np.random.uniform(120, 160, n_samples),
        'accelerations': np.random.uniform(0, 0.02, n_samples),
        'fetal_movement': np.random.uniform(0, 0.5, n_samples),
        'uterine_contractions': np.random.uniform(0, 0.02, n_samples),
        'light_decelerations': np.random.uniform(0, 0.01, n_samples),
        'severe_decelerations': np.random.uniform(0, 0.001, n_samples),
        'prolongued_decelerations': np.random.uniform(0, 0.001, n_samples),
        'abnormal_short_term_variability': np.random.uniform(20, 80, n_samples),
        'mean_value_of_short_term_variability': np.random.uniform(2, 10, n_samples),
        'percentage_of_time_with_abnormal_long_term_variability': np.random.uniform(0, 90, n_samples),
        'mean_value_of_long_term_variability': np.random.uniform(10, 50, n_samples),
        'histogram_width': np.random.uniform(30, 100, n_samples),
        'histogram_min': np.random.uniform(50, 120, n_samples),
        'histogram_max': np.random.uniform(150, 200, n_samples),
        'histogram_number_of_peaks': np.random.randint(1, 10, n_samples),
        'histogram_number_of_zeroes': np.random.randint(0, 5, n_samples),
        'histogram_mode': np.random.uniform(120, 160, n_samples),
        'histogram_mean': np.random.uniform(120, 160, n_samples),
        'histogram_median': np.random.uniform(120, 160, n_samples),
        'histogram_variance': np.random.uniform(50, 200, n_samples),
        'histogram_tendency': np.random.uniform(-1, 1, n_samples),
        'fetal_health': np.random.choice([1, 2, 3], n_samples, p=[0.78, 0.14, 0.08])
    })

    # Save the sample data as cache
    data.to_csv('fetal_health_cache.csv', index=False)
    return data


def process_data(X, y):
    columns_names = list(X.columns)
    scaler = preprocessing.StandardScaler()
    X_df = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_df, columns=columns_names)

    X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)

    y_train = y_train - 1
    y_test = y_test - 1
    return X_train, X_test, y_train, y_test


def create_model(X):
    reset_seeds()
    model = Sequential()
    model.add(InputLayer(input_shape=(X.shape[1], )))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def config_mlflow():
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'renansantosmendes'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '6d730ef4a90b1caf28fbb01e5748f0874fda6077'
    mlflow.set_tracking_uri('https://dagshub.com/renansantosmendes/puc_lectures_mlops.mlflow')

    mlflow.keras.autolog(log_models=True,
                         log_input_examples=True,
                         log_model_signatures=True)


def train_model(model, X_train, y_train, is_train=True):
    with mlflow.start_run(run_name='experiment_mlops_ead_slompo') as _:
        model.fit(X_train,
                  y_train,
                  epochs=50,
                  validation_split=0.2,
                  verbose=3)


if __name__ == '__main__':
    X, y = read_data()
    X_train, X_test, y_train, y_test = process_data(X, y)
    model = create_model(X)
    config_mlflow()
    train_model(model, X_train, y_train)
