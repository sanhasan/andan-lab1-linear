from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.download import load_images
from utils.feature_selection import select_features


def prepare_features(X_train, X_test, y_train, features_count):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # нормализует по данным с тренировочной

    selected = select_features(X_train, y_train, features_count)
    X_train = X_train[:, selected]
    X_test = X_test[:, selected]
    return X_train, X_test, selected


def prepare_dataset(
    parquet_paths,
    output_dir,
    image_size,
    test_size,
    random_state,
    features_count,
):
    X, y = load_images(parquet_paths, image_size)
    save_all_features(output_dir, X, y)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size,
        random_state,
    )

    X_train, X_test, features_indexes = prepare_features(
        X_train, X_test, y_train, features_count
    )

    save_split_dataset(output_dir, X_train, y_train, X_test, y_test)

    return {
        "all_samples": len(y),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "classes": {-1: "фотография", 1: "аниме"},
        "features_indexes": features_indexes,
    }



def split_dataset(X, y, test_size, random_state):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # чтобы равномерно по категориям разделилось
    )


def save_all_features(output_dir, X, y):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "all_features.npz", X=X, y=y)


def save_split_dataset(output_dir, X_train, y_train, X_test, y_test):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(output_dir / "test.npz", X=X_test, y=y_test)
