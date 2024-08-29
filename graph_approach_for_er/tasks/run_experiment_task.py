import hashlib
import logging
import os
import random
import time

import joblib
import luigi
import numpy as np
import pandas as pd
import torch

from graph_approach_for_er.dataloader.magellan_bert_loader import MagellanBertLoader
from graph_approach_for_er.dataloader.magellan_non_bert_loader import MagellanNonBertLoader
from graph_approach_for_er.dataloader.mp_bert_loader import MarktPilotBertLoader
from graph_approach_for_er.dataloader.mp_non_bert_loader import MarktPilotNonBertLoader
from graph_approach_for_er.dataloader.wdc_bert_loader import WdcBertLoader
from graph_approach_for_er.dataloader.wdc_non_bert_loader import WdcNonBertLoader
from graph_approach_for_er.metrics.metricsbag import MetricsBag
from graph_approach_for_er.models.bert import get_model as get_bert_model
from graph_approach_for_er.tasks.base import LuigiBaseTask
from graph_approach_for_er.tasks.preprocess_magellan_train_data import PreprocessMagellanTrainDataTask
from graph_approach_for_er.tasks.preprocess_magellan_val_test_data import PreprocessMagellanValTestDataTask
from graph_approach_for_er.tasks.preprocess_mp_train_data import PreprocessMarktPilotTrainDataTask
from graph_approach_for_er.tasks.preprocess_mp_val_test_data import PreprocessMarktPilotValTestDataTask
from graph_approach_for_er.tasks.preprocess_wdc_train_data import PreprocessWDCTrainDataTask
from graph_approach_for_er.tasks.preprocess_wdc_val_test_data import PreprocessWdcValTestDataTask
from graph_approach_for_er.tasks.train_ditto import TrainDittoTask
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

        if self.experiment.dataset == "markt_pilot":
            requirements["train_data"] = PreprocessMarktPilotTrainDataTask(experiment_name=self.experiment_name)
            requirements["val_data"] = PreprocessMarktPilotValTestDataTask(experiment_name=self.experiment_name)
        elif self.experiment.dataset == "wdc":
            requirements["train_data"] = PreprocessWDCTrainDataTask(experiment_name=self.experiment_name)
            requirements["val_data"] = PreprocessWdcValTestDataTask(experiment_name=self.experiment_name)
        elif self.experiment.dataset in ["amazon_google", "walmart_amazon"]:
            requirements["train_data"] = PreprocessMagellanTrainDataTask(experiment_name=self.experiment_name)
            requirements["val_data"] = PreprocessMagellanValTestDataTask(experiment_name=self.experiment_name)

        return requirements

    def run(self) -> None:
        # fix randomness
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)

        start_time = time.time()

        if self.experiment.model == "ditto":
            # make sure to actually rerun the task
            task = TrainDittoTask(experiment_name=self.experiment_name)
            task.invalidate()
            luigi.build([TrainDittoTask(experiment_name=self.experiment_name)], local_scheduler=True)
            ditto_path = TrainDittoTask(experiment_name=self.experiment_name).output().path

            ditto_output = joblib.load(ditto_path)
            probabilities = ditto_output["probabilities"]
            epochs_trained = ditto_output["epochs_trained"]
        else:
            probabilities, epochs_trained = self.run_task()

        test_input_path = self.input()["val_data"].path.replace("prefix_", "custom_test_").replace("_suffix",
                                                                                                   ".parquet")
        df_test = pd.read_parquet(test_input_path)
        true_labels = df_test.label.to_list()

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

    def run_task(self):
        # get data
        train_input_path = self.input()["train_data"].path.replace("prefix_", "custom_train_").replace("_suffix",
                                                                                                       ".parquet")
        df_train = pd.read_parquet(train_input_path)

        val_input_path = self.input()["val_data"].path.replace("prefix_", "custom_val_").replace("_suffix", ".parquet")
        df_val = pd.read_parquet(val_input_path)

        test_input_path = self.input()["val_data"].path.replace("prefix_", "custom_test_").replace("_suffix",
                                                                                                   ".parquet")
        df_test = pd.read_parquet(test_input_path)

        if self.experiment.dataset == "markt_pilot":
            if self.experiment.model == "bert":
                loader_factory = MarktPilotBertLoader(df_val, df_test, self.experiment)
            else:
                loader_factory = MarktPilotNonBertLoader(df_train, df_val, df_test, self.experiment)

        elif self.experiment.dataset == "wdc":
            if self.experiment.model == "bert":
                loader_factory = WdcBertLoader(df_val, df_test, self.experiment)
            else:
                loader_factory = WdcNonBertLoader(df_train, df_val, df_test, self.experiment)

        elif self.experiment.dataset in ["amazon_google", "walmart_amazon"]:
            if self.experiment.model == "bert":
                loader_factory = MagellanBertLoader(df_val, df_test, self.experiment)
            else:
                loader_factory = MagellanNonBertLoader(df_train, df_val, df_test, self.experiment)

        else:
            raise ValueError("Unknown dataset name.")

        val_loader = loader_factory.get_val_loader()
        test_loader = loader_factory.get_test_loader()

        # get model
        if self.experiment.model == "bert":
            model = get_bert_model(self.experiment)
        else:
            raise ValueError("Unknown model name.")

        # get trainer
        if self.experiment.model == "bert":
            trainer = BertTrainer(
                model=model,
                val_dataloader=val_loader,
                experiment=self.experiment,
                working_dir=self.global_config.working_dir,
            )
        else:
            raise ValueError("Unknown model name when trying to select trainer.)")

        # run
        trainer.train(df_train)

        epochs_trained = trainer.training_stats[-1]["epoch"]
        probabilities, _ = trainer.evaluate(test_loader)

        return probabilities, epochs_trained
