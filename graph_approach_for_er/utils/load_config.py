import yaml
import importlib.resources as pkg_resources
from dataclasses import dataclass

import graph_approach_for_er
import graph_approach_for_er.experiments


@dataclass(frozen=True, eq=True)
class ExperimentConfiguration:
    name: str
    dataset: str
    path_to_train_set: str
    path_to_val_set: str
    path_to_test_set: str
    model: str
    offline_augmentation: list  # synonym, swap_random, insert_random, delete_random
    online_augmentation: list  # mixup
    batch_size: int
    max_string_len: int
    max_input_length: int
    epochs: int
    label_noise_min_degree: int
    label_noise_threshold: int
    pos_neg_ratio_cap: int

    def print_data(obj):
        output = ""
        max_field_length = max(len(field.name) for field in obj.__dataclass_fields__.values())

        for field in obj.__dataclass_fields__.values():
            output += f"{field.name.ljust(max_field_length+1)} {getattr(obj, field.name)}\n"

        return output


@dataclass
class GlobalConfiguration:
    working_dir: str


def load_config(name):
    with pkg_resources.open_text(graph_approach_for_er.experiments, f"{name}.yml") as config_file:
        config = yaml.safe_load(config_file)

    config["name"] = name
    return ExperimentConfiguration(**config)


def load_global_config():
    with pkg_resources.open_text(graph_approach_for_er, "config.yml") as config_file:
        config = yaml.safe_load(config_file)

    return GlobalConfiguration(**config)
