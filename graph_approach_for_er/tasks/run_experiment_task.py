import hashlib
import logging
import os
import random
import time

import luigi
import numpy as np
import pandas as pd
import torch

from graph_approach_for_er.dataloader.mrsp_loader import MrspLoader
from graph_approach_for_er.metrics.metricsbag import MetricsBag
from graph_approach_for_er.models.bert import get_model as get_bert_model
from graph_approach_for_er.tasks.base import LuigiBaseTask
from graph_approach_for_er.trainer.bert_trainer import Trainer as BertTrainer
from graph_approach_for_er.utils.load_config import load_global_config, load_config


class RunExperimentTask(LuigiBaseTask):
    experiment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_config = load_global_config()
        self.experiment = load_config(self.experiment_name)

        relevant_fields = self.experiment.__str__()
        self.filename = hashlib.md5(relevant_fields.encode()).hexdigest()
        self.output_path = os.path.join(self.global_config.working_dir, "experiment_results")

    def requires(self):
        requirements = {}
        return requirements

    def run(self) -> None:
        # fix randomness
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)

        start_time = time.time()

        df_test = self.get_df_test()
        true_labels = df_test.label.to_list()

        probabilities, epochs_trained = self.run_task(df_test)

        try:
            bag = MetricsBag(y=true_labels, y_hat=probabilities[:, 1])
        except:
            bag = MetricsBag(y=true_labels, y_hat=probabilities)
        bag.evaluate()

        end_time = time.time()
        duration_in_sec = end_time - start_time
        duration_in_min = round(duration_in_sec / 60)
        duration_string = f"Execution took {duration_in_min} minutes"

        info_string = f"Max. F1-score: {round(bag.f1_scores[bag.index_of_maximum_f1_score], 4)}"
        logging.info(info_string)
        logging.info(duration_string)

        self.write(
            "\n".join([info_string, self.experiment.print_data(), f"Epochs actually trained: {epochs_trained}",
                       duration_string])
        )

    def output(self) -> luigi.LocalTarget:
        return self.make_output_target(self.output_path, f"{self.experiment.name}_{self.filename}.txt")

    def get_df_test(self):
        if self.experiment.dataset == "mrsp":
            df_test = pd.read_parquet(os.path.join(self.global_config.working_dir, self.experiment.path_to_test_set))
            str_cols = ["sentence1", "sentence2"]
            df_test[str_cols] = df_test[str_cols].astype(str)
            df_test["label"] = df_test["label"].astype(int)
            return df_test

        raise ValueError("Unknown dataset")

    def run_task(self, df_test):
        if self.experiment.dataset == "mrsp":
            str_cols = ["sentence1", "sentence2"]

            df_train = pd.read_parquet(os.path.join(self.global_config.working_dir, self.experiment.path_to_train_set))
            df_train[str_cols] = df_train[str_cols].astype(str)
            df_train["label"] = df_train["label"].astype(int)

            df_val = pd.read_parquet(os.path.join(self.global_config.working_dir, self.experiment.path_to_val_set))
            df_val[str_cols] = df_val[str_cols].astype(str)
            df_val["label"] = df_val["label"].astype(int)

            loader_factory = MrspLoader(df_train=df_train, df_val=df_val, df_test=df_test, experiment=self.experiment)

        else:
            raise ValueError("Unknown dataset name.")

        train_loader = loader_factory.get_train_loader()
        val_loader = loader_factory.get_val_loader()
        test_loader = loader_factory.get_test_loader()

        model = get_bert_model(self.experiment)

        # get trainer
        trainer = BertTrainer(
            model=model,
            val_dataloader=val_loader,
            experiment=self.experiment,
            working_dir=self.global_config.working_dir,
            train_dataloader=train_loader
        )

        # run
        if len(self.experiment.online_augmentation) == 0:
            trainer.train_baseline()
        else:
            trainer.train(df_train)

        epochs_trained = trainer.training_stats[-1]["epoch"]
        probabilities, _ = trainer.evaluate(test_loader)

        return probabilities, epochs_trained
