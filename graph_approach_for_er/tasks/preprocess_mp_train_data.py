import luigi
import os
import pandas as pd
import hashlib

from graph_approach_for_er.tasks.base import LuigiBaseTask
from graph_approach_for_er.utils.load_config import load_global_config, load_config
from graph_approach_for_er.utils.mp_dataset import set_datatypes_and_limit_string_length
from graph_approach_for_er.utils.ditto import write_mp_data_in_ditto_format
from graph_approach_for_er.augmenter.offline_augmentation import apply_offline_augmentation


class PreprocessMarktPilotTrainDataTask(LuigiBaseTask):
    experiment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_config = load_global_config()
        self.experiment = load_config(self.experiment_name)

        # the relevant fields of the config get hashed, to detect unchanged fields in case of a rerun and save
        # computation time
        relevant_fields = " ".join([
            self.experiment.path_to_train_set,
            self.experiment.offline_augmentation.__str__()
        ])
        self.filename = hashlib.md5(relevant_fields.encode()).hexdigest()
        self.output_path = os.path.join(self.global_config.working_dir, "data", "intermediate", "load_mp_data")

    def run(self) -> None:
        df_train = pd.read_parquet(os.path.join(self.global_config.working_dir, self.experiment.path_to_train_set))
        df_train = set_datatypes_and_limit_string_length(df_train, self.experiment)

        # do augmentation on text
        if self.experiment.offline_augmentation is not None and len(self.experiment.offline_augmentation) > 0:
            columns_to_augment = [
                "searched_brand", "searched_number", "searched_name", "searched_description", "searched_group",
                "found_brand", "found_number", "found_name", "found_description", "found_group"]
            df_train = apply_offline_augmentation(
                df_train, columns_to_augment, self.experiment.offline_augmentation,
                experiment_configuration=self.experiment, global_configuration=self.global_config)

        df_train.to_parquet(os.path.join(self.output_path, f"custom_train_{self.filename}.parquet"))

        # save in ditto format
        write_mp_data_in_ditto_format(df_train, os.path.join(self.output_path, f"ditto_train_{self.filename}"))

        self.write("")

    def output(self) -> luigi.LocalTarget:
        return self.make_output_target(self.output_path, f"prefix_{self.filename}_suffix")
