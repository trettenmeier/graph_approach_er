import os
from dataclasses import dataclass
import json
import yaml


@dataclass
class Dataset:
    name: str
    path_to_train_set: str
    path_to_val_set: str
    path_to_test_set: str


DATASETS = [
    Dataset(
        name="amazon_google",
        path_to_train_set="data/raw/amazon_google/train.csv",
        path_to_val_set="data/raw/amazon_google/valid.csv",
        path_to_test_set="data/raw/amazon_google/test.csv",
    ),
    Dataset(
        name="walmart_amazon",
        path_to_train_set="data/raw/walmart_amazon/train.csv",
        path_to_val_set="data/raw/walmart_amazon/valid.csv",
        path_to_test_set="data/raw/walmart_amazon/test.csv",
    ),
    Dataset(
        name="markt_pilot",
        path_to_train_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_train.parquet",
        path_to_val_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_val.parquet",
        path_to_test_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_test.parquet",
    ),
    Dataset(
        name="markt_pilot",
        path_to_train_set="data/raw/markt_pilot_dataset/l_markt_pilot_dataset_train.parquet",
        path_to_val_set="data/raw/markt_pilot_dataset/l_markt_pilot_dataset_val.parquet",
        path_to_test_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_test.parquet",
    ),
    Dataset(
        name="markt_pilot",
        path_to_train_set="data/raw/markt_pilot_dataset/m_markt_pilot_dataset_train.parquet",
        path_to_val_set="data/raw/markt_pilot_dataset/m_markt_pilot_dataset_val.parquet",
        path_to_test_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_test.parquet",
    ),
    Dataset(
        name="markt_pilot",
        path_to_train_set="data/raw/markt_pilot_dataset/s_markt_pilot_dataset_train.parquet",
        path_to_val_set="data/raw/markt_pilot_dataset/s_markt_pilot_dataset_val.parquet",
        path_to_test_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_test.parquet",
    ),
    Dataset(
        name="wdc",
        path_to_train_set="data/raw/wdc_lspm/all_train/all_train_xlarge.json.gz",
        path_to_val_set="data/raw/wdc_lspm/all_valid/all_valid_xlarge.csv",
        path_to_test_set="data/raw/wdc_lspm/all_gs.json.gz",
    ),
    Dataset(
        name="wdc",
        path_to_train_set="data/raw/wdc_lspm/all_train/all_train_large.json.gz",
        path_to_val_set="data/raw/wdc_lspm/all_valid/all_valid_large.csv",
        path_to_test_set="data/raw/wdc_lspm/all_gs.json.gz",
    ),
    Dataset(
        name="wdc",
        path_to_train_set="data/raw/wdc_lspm/all_train/all_train_medium.json.gz",
        path_to_val_set="data/raw/wdc_lspm/all_valid/all_valid_medium.csv",
        path_to_test_set="data/raw/wdc_lspm/all_gs.json.gz",
    ),
    Dataset(
        name="wdc",
        path_to_train_set="data/raw/wdc_lspm/all_train/all_train_small.json.gz",
        path_to_val_set="data/raw/wdc_lspm/all_valid/all_valid_small.csv",
        path_to_test_set="data/raw/wdc_lspm/all_gs.json.gz",
    ),
]

MODELS = ["bert", "ditto"]

OFFLINE_AUGMENTATION = []
ONLINE_AUGMENTATION = ["graph"]
NO_AUGMENTATION = ["no_aug"]


def create_yml(dataset: Dataset, model: str, augmentation: str, is_offline: bool, path,
               label_noise_min_degree: int, label_noise_threshold: int, pos_neg_ratio_cap: int):
    def get_subset_size(path):
        if "wdc" in path and "_xlarge" in path:
            return "_xlarge"
        if "wdc" in path and "_large" in path:
            return "_large"
        if "wdc" in path and "_medium" in path:
            return "_medium"
        if "wdc" in path and "_small" in path:
            return "_small"
        if "markt_pilot" in path and "l_" in path:
            return "_l"
        if "markt_pilot" in path and "m_" in path:
            return "_m"
        if "markt_pilot" in path and "s_" in path:
            return "_s"
        if "markt_pilot_dataset_train" in path:
            return "_full"

        return ""

    epochs = 30
    dataset_full_name = f"{dataset.name}{get_subset_size(dataset.path_to_train_set)}"

    if "_full" in dataset_full_name:
        epochs = 2
    elif "_xlarge" in dataset_full_name:
        epochs = 10
    elif "_l" in dataset_full_name:
        epochs = 10
    elif "_m" in dataset_full_name:
        epochs = 10
    elif "_large" in dataset_full_name:
        epochs = 20

    yml_data = {
        "model": model,
        "dataset": dataset.name,
        "path_to_train_set": dataset.path_to_train_set,
        "path_to_val_set": dataset.path_to_val_set,
        "path_to_test_set": dataset.path_to_test_set,
        "batch_size": 32,
        "max_string_len": 1000,
        "max_input_length": 256 if model == "ditto" else 128,
        "epochs": epochs,
        "label_noise_min_degree": label_noise_min_degree,
        "label_noise_threshold": label_noise_threshold,
        "pos_neg_ratio_cap": pos_neg_ratio_cap,
    }

    if is_offline:
        if augmentation == "no_aug":
            yml_data["offline_augmentation"] = []
        else:
            yml_data["offline_augmentation"] = [augmentation]
        yml_data["online_augmentation"] = []
    else:
        yml_data["offline_augmentation"] = []
        yml_data["online_augmentation"] = [augmentation]

    filename = f"{dataset.name}{get_subset_size(dataset.path_to_train_set)}_{model}_{str(hash(json.dumps(yml_data)))}.yml"

    with open(os.path.join(path, filename), "w") as file:
        yaml.dump(yml_data, file)


def main():
    path = os.path.join("..", "experiments")
    os.makedirs(path, exist_ok=True)

    label_noise_min_degrees = [2, 5, 7]
    label_noise_thresholds = [3, 5, 7]
    pos_neg_ratio_caps = [2, 5, 8]

    for dataset in DATASETS:
        for model in MODELS:
            for label_noise_min_degree in label_noise_min_degrees:
                for label_noise_threshold in label_noise_thresholds:
                    for pos_neg_ratio_cap in pos_neg_ratio_caps:
                        for online_augmentation in ONLINE_AUGMENTATION:
                            create_yml(
                                dataset,
                                model,
                                online_augmentation,
                                is_offline=False,
                                path=path,
                                label_noise_min_degree=label_noise_min_degree,
                                label_noise_threshold=label_noise_threshold,
                                pos_neg_ratio_cap=pos_neg_ratio_cap
                            )


if __name__ == "__main__":
    main()
