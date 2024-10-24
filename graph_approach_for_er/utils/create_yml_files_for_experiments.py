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
        name="mrsp",
        path_to_train_set="data/raw/beyond_er/mrsp/train-00000-of-00001.parquet",
        path_to_val_set="data/raw/beyond_er/mrsp/validation-00000-of-00001.parquet",
        path_to_test_set="data/raw/beyond_er/mrsp/test-00000-of-00001.parquet",
    )
]

ONLINE_AUGMENTATION = ["graph"]


def create_yml(path: str, dataset: Dataset, label_noise_min_degree: int, label_noise_threshold: int,
               pos_neg_ratio_cap: int):
    epochs = 30

    yml_data = {
        "model": "bert",
        "dataset": dataset.name,
        "path_to_train_set": dataset.path_to_train_set,
        "path_to_val_set": dataset.path_to_val_set,
        "path_to_test_set": dataset.path_to_test_set,
        "batch_size": 32,
        "max_string_len": 1000,
        "max_input_length": 128,
        "epochs": epochs,
        "online_augmentation": ["graph"],
        "offline_augmentation": [],
        "label_noise_min_degree": label_noise_min_degree,
        "label_noise_threshold": label_noise_threshold,
        "pos_neg_ratio_cap": pos_neg_ratio_cap,
    }

    filename = f"{dataset.name}_{str(hash(json.dumps(yml_data)))}.yml"

    with open(os.path.join(path, filename), "w") as file:
        yaml.dump(yml_data, file)


def main():
    path = os.path.join("..", "experiments")
    os.makedirs(path, exist_ok=True)

    label_noise_min_degrees = [2, 5, 7]
    label_noise_thresholds = [3, 5, 7]
    pos_neg_ratio_caps = [2, 5, 8]

    for dataset in DATASETS:
        for label_noise_min_degree in label_noise_min_degrees:
            for label_noise_threshold in label_noise_thresholds:
                for pos_neg_ratio_cap in pos_neg_ratio_caps:
                    create_yml(
                        dataset=dataset,
                        path=path,
                        label_noise_min_degree=label_noise_min_degree,
                        label_noise_threshold=label_noise_threshold,
                        pos_neg_ratio_cap=pos_neg_ratio_cap
                    )


if __name__ == "__main__":
    main()
