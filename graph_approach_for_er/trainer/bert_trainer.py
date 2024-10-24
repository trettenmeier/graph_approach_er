import datetime
import logging
import os
import random
import time
from copy import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from graph_approach_for_er.graph_augmentation.graph_augmenter import GraphAugmentation
from graph_approach_for_er.models.bert import get_model
from graph_approach_for_er.utils.early_stopping import EarlyStopper, StopTrainingWhenTrainLossIsNearZero
from graph_approach_for_er.utils.load_config import ExperimentConfiguration


class Trainer:
    def __init__(self, model, val_dataloader, experiment: ExperimentConfiguration, working_dir: str,
                 train_dataloader=None):
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info(f"Using GPU:, {torch.cuda.get_device_name(0)}")
        else:
            logging.info("No GPU available")
            self.device = torch.device("cpu")

        self.experiment = experiment
        self.model_name = experiment.model
        self.model = model
        self.model.to(self.device)

        self.early_stopper = EarlyStopper()
        self.stop_training = False

        self.model_path = os.path.join(working_dir, "models", self.model_name, "trained_model")
        os.makedirs(self.model_path, exist_ok=True)

        self.epochs = experiment.epochs
        self.val_dataloader = val_dataloader
        self.train_dataloader = train_dataloader
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

        self.reduce_lr_on_plateau = ReduceLROnPlateau(self.optimizer, "min", factor=0.5, patience=5, verbose=True)

        self.training_stats = []
        self.best_val_f1 = -1

    def load_trained_model(self):
        self.model = get_model(self.experiment)
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, "model.pt")))
        self.model.to(self.device)
        self.model.eval()

    def move_to_cuda_and_get_model_output(self, batch, train=False):
        input_ids = batch["input_ids"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["labels"].to(self.device, dtype=torch.long)

        # forward pass
        if train:
            result, labels = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
        else:
            with torch.no_grad():
                result = self.model(input_ids=input_ids, token_type_ids=token_type_ids)

        return result, labels

    def train_baseline(self):
        self.training_stats = []
        training_start_time = time.time()

        loss = CrossEntropyLoss()

        logging.info(f"======== Starting baseline training ========")
        t0 = time.time()
        total_train_loss = 0
        avg_train_loss = 0

        for epoch in range(0, self.epochs):
            logging.info(f"======== Epoch {epoch + 1} / {self.epochs} ========")

            self.model.train()

            if StopTrainingWhenTrainLossIsNearZero.training_loss_is_near_zero(avg_train_loss) and epoch != 0:
                logging.info("stopping training because loss is near zero")
                logging.info("")
                logging.info("Loading best model.")
                self.load_trained_model()
                self.model.eval()
                logging.info(f"Total training took {self.format_time(time.time() - training_start_time)}")
                break

            for step, batch in enumerate(self.train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    logging.info(f"Step {step}. Elapsed: {elapsed}")

                self.model.zero_grad()
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device, dtype=torch.long)

                # forward pass
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
                logits = outputs["logits"]
                ce_loss = loss(logits, labels.to(self.device))
                total_train_loss += ce_loss.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                ce_loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
                loss.zero_grad()

                # validate every n steps
                if step % 10000 == 0 and not step == 0:
                    self.validation(0, 0, self.format_time(time.time() - t0))
                    self.model.train()

            avg_train_loss = total_train_loss / step
            training_time = self.format_time(time.time() - t0)

            logging.info("")
            logging.info(f"  Average training loss: {avg_train_loss}")
            logging.info(f"  Training epoch took: {training_time}")
            logging.info("")
            logging.info("Running Validation...")

            avg_val_loss = self.validation(epoch, avg_train_loss, training_time)

            self.reduce_lr_on_plateau.step(avg_val_loss)

            if self.stop_training:
                logging.info("early stopping.")
                break

        logging.info("")
        logging.info("Loading best model.")
        self.load_trained_model()
        logging.info(f"Total training took {self.format_time(time.time() - training_start_time)}")

    def train(self, df_train):
        self.training_stats = []
        training_start_time = time.time()

        loss = CrossEntropyLoss()

        if self.experiment.dataset == "wdc":
            cols_left = ["title_left", "brand_left", "category_left", "description_left"]
            cols_right = ["title_right", "brand_right", "category_right", "description_right"]
        elif self.experiment.dataset == "markt_pilot":
            cols_left = ["searched_brand", "searched_number", "searched_name", "searched_group", "searched_description"]
            cols_right = ["found_brand", "found_number", "found_name", "found_group", "found_description"]
        elif self.experiment.dataset in ["amazon_google", "walmart_amazon"]:
            cols = df_train.columns.tolist()
            cols_left = [i for i in cols if "left" in i]
            cols_right = [i for i in cols if "right" in i]
            df_train[cols_left] = df_train[cols_left].astype(str)
            df_train[cols_right] = df_train[cols_right].astype(str)
        elif self.experiment.dataset == "mrsp":
            cols_left = ["sentence1"]
            cols_right = ["sentence2"]
            df_train[cols_left] = df_train[cols_left].astype(str)
            df_train[cols_right] = df_train[cols_right].astype(str)

        else:
            raise NotImplementedError

        graph_augmentation = GraphAugmentation(
            df_train=df_train, cols_left=cols_left, cols_right=cols_right, label="label", experiment=self.experiment
        )

        # initial training phase
        logging.info(f"======== Initial Training Phase ========")
        t0 = time.time()
        self.model.train()
        total_train_loss = 0

        for step, batch in enumerate(graph_augmentation.all_pairs_dataloader_for_initial_training()):
            if step % 40 == 0 and not step == 0:
                elapsed = self.format_time(time.time() - t0)
                logging.info(f"Step {step}. Elapsed: {elapsed}")

            self.model.zero_grad()
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels = batch["labels"].to(self.device, dtype=torch.long)

            # forward pass
            outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
            logits = outputs["logits"]
            ce_loss = loss(logits, labels.to(self.device))
            total_train_loss += ce_loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            ce_loss.backward()
            self.optimizer.step()
            self.model.zero_grad()
            loss.zero_grad()

            # validate every n steps
            if step % 10000 == 0 and not step == 0:
                self.validation(0, 0, self.format_time(time.time() - t0))
                self.model.train()

        avg_train_loss = total_train_loss / step
        training_time = self.format_time(time.time() - t0)

        logging.info("")
        logging.info(f"  Average training loss: {avg_train_loss}")
        logging.info(f"  Training epoch took: {training_time}")
        logging.info("")
        logging.info("Running Validation...")

        _ = self.validation(0, avg_train_loss, training_time)

        # hard examples training phase
        logging.info(f"======== Starting hard examples training phase ========")

        for epoch in range(0, self.epochs):
            logging.info(f"======== Epoch {epoch + 1} / {self.epochs} ========")
            t0 = time.time()
            self.model.train()

            if StopTrainingWhenTrainLossIsNearZero.training_loss_is_near_zero(avg_train_loss) and epoch != 0:
                logging.info("stopping training because loss is near zero")
                logging.info("")
                logging.info("Loading best model.")
                self.load_trained_model()
                self.model.eval()
                logging.info(f"Total training took {self.format_time(time.time() - training_start_time)}")
                break

            total_train_loss = 0

            list_with_hardness = []

            with torch.no_grad():
                logging.info("feeding forward all pairs to determine the loss")
                for batch in tqdm(
                        graph_augmentation.all_pairs_dataloader(),
                        total=int(len(graph_augmentation.all_pairs) / self.experiment.batch_size),
                ):
                    input_ids = []
                    token_type_ids = []
                    labels = []

                    for left_index, right_index, label in batch:
                        current_input_ids, current_token_type_ids = graph_augmentation.construct_single_data_point(
                            left_index, right_index
                        )
                        input_ids.append(current_input_ids)
                        token_type_ids.append(current_token_type_ids)
                        labels.append(label)

                    input_ids = torch.stack(input_ids)
                    token_type_ids = torch.stack(token_type_ids)
                    labels = torch.tensor(labels)

                    input_ids = input_ids.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    labels = labels.to(self.device, dtype=torch.long)

                    outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)

                    logits = outputs["logits"]

                    ce_loss = F.cross_entropy(logits, labels, reduction="none")

                    # put everything back together
                    for i in range(0, len(batch)):
                        list_with_hardness.append((batch[i][0], batch[i][1], batch[i][2], ce_loss[i].detach().item()))

            # remove nodes that are likely label noise
            significant_nodes = GraphAugmentation.find_label_noise(
                list_with_hardness=list_with_hardness,
                threshold=graph_augmentation.label_noise_threshold,
                min_degree_to_consider=graph_augmentation.label_noise_min_degree
            )

            list_with_hardness_cleaned = [i for i in list_with_hardness if
                                          i[0] not in significant_nodes and i[1] not in significant_nodes]
            logging.info(
                f"Removed {len(list_with_hardness) - len(list_with_hardness_cleaned)} of {len(list_with_hardness)} entries")

            list_with_hardness_cleaned = sorted(list_with_hardness_cleaned, key=lambda x: x[3], reverse=True)
            positives_list = [i for i in list_with_hardness_cleaned if i[2] == 1]
            negatives_list = [i for i in list_with_hardness_cleaned if i[2] == 0]

            # try this: remove the hardest 1% of the lists as the probability for label noise is highest
            positives_list = positives_list[20:]

            negatives_list = sorted(negatives_list, key=lambda x: x[3], reverse=True)
            negatives_list = negatives_list[20:]

            final_list = copy(positives_list)

            # select amount to train on
            pos_neg_ratio = df_train[df_train.label == 1].shape[0] / df_train[df_train.label == 0].shape[0]

            negatives_list = sorted(negatives_list, key=lambda x: x[3], reverse=True)
            expand_factor = min([int(1 / pos_neg_ratio), graph_augmentation.pos_neg_ratio_cap])
            final_list.extend(negatives_list[: len(positives_list) * expand_factor])

            random.shuffle(final_list)

            for step, batch in enumerate(graph_augmentation.batch_iterable(final_list)):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    logging.info(
                        f"Step {step}  of  {int(len(final_list) / self.experiment.batch_size)}. Elapsed: {elapsed}")

                self.model.zero_grad()
                self.optimizer.zero_grad()

                input_ids = []
                token_type_ids = []
                labels = []

                for left_index, right_index, label, calculated_loss in batch:
                    left = graph_augmentation.get_datapoint_from_table_idx(left_index)
                    right = graph_augmentation.get_datapoint_from_table_idx(right_index)

                    current_input_ids, current_token_type_ids = graph_augmentation.tokenize(left, right)
                    input_ids.append(current_input_ids)
                    token_type_ids.append(current_token_type_ids)
                    labels.append(label)

                input_ids = torch.stack(input_ids)
                token_type_ids = torch.stack(token_type_ids)
                labels = torch.tensor(labels)

                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)

                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)

                logits = outputs["logits"]

                ce_loss = loss(logits, labels)

                total_train_loss += ce_loss.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                ce_loss.backward()
                self.optimizer.step()
                self.model.zero_grad()
                loss.zero_grad()

                # validate every n steps
                if step % 10000 == 0 and not step == 0:
                    self.validation(epoch, 0, self.format_time(time.time() - t0))
                    self.model.train()

            avg_train_loss = total_train_loss / (len(list_with_hardness) / self.experiment.batch_size)
            training_time = self.format_time(time.time() - t0)

            logging.info("")
            logging.info(f"  Average training loss: {avg_train_loss}")
            logging.info(f"  Training epoch took: {training_time}")
            logging.info("")
            logging.info("Running Validation...")
            avg_val_loss = self.validation(epoch, avg_train_loss, training_time)

            self.reduce_lr_on_plateau.step(avg_val_loss)

            if self.stop_training:
                logging.info("early stopping.")
                break

        logging.info("")
        logging.info("Loading best model.")
        self.load_trained_model()
        logging.info(f"Total training took {self.format_time(time.time() - training_start_time)}")

    def validation(self, epoch, avg_train_loss, training_time) -> float:
        t0 = time.time()
        total_eval_loss = 0
        all_logits = []
        all_labels = []

        self.model.eval()
        from torch.nn import CrossEntropyLoss

        loss = CrossEntropyLoss()

        for batch in self.val_dataloader:
            output, labels = self.move_to_cuda_and_get_model_output(batch, train=False)
            logits = output["logits"]

            the_loss = loss(logits, labels)
            total_eval_loss += the_loss.item()

            loss.zero_grad()
            raw_logits = logits.detach().to("cpu").numpy()
            softmaxed_logits = softmax(raw_logits, axis=1)

            all_logits.extend(softmaxed_logits[:, 1])

            # create integer labels for f1 score
            int_labels = labels.to("cpu").to(torch.int64)
            all_labels.extend(int_labels)

        f1 = self.compute_f1(all_logits, all_labels)
        logging.info(f"  Validation F1: {f1}")

        avg_val_loss = total_eval_loss / len(self.val_dataloader)
        validation_time = self.format_time(time.time() - t0)

        logging.info(f"  Validation Loss: {avg_val_loss}")
        logging.info(f"  Validation took: {validation_time}")

        self.training_stats.append(
            {
                "epoch": epoch + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. F1.": f1,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

        # save best model
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pt"))

        # early stopping
        self.stop_training = self.early_stopper.early_stop(avg_val_loss)

        return avg_val_loss

    def format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def compute_f1(self, probability_pos, labels):
        thresholds = np.linspace(0, 1, num=50)

        def calc_func(e: float):
            predictions = [1 if x > e else 0 for x in probability_pos]
            return f1_score(labels, predictions)

        results = np.array([calc_func(thresh) for thresh in thresholds])

        return max(results)

    def evaluate(self, test_dataloader):
        self.load_trained_model()
        self.model.eval()

        predictions, true_labels = [], []

        for batch in test_dataloader:
            output, labels = self.move_to_cuda_and_get_model_output(batch, train=False)
            logits = output["logits"]

            logits = logits.detach().cpu().numpy()
            label_ids = labels.to("cpu").numpy()

            predictions.append(logits)
            true_labels.append(label_ids)

        pred_flat = list(np.concatenate(predictions))
        pred_flat = softmax(pred_flat, axis=1)

        true_labels_flat = list(np.concatenate(true_labels))

        return pred_flat, true_labels_flat
