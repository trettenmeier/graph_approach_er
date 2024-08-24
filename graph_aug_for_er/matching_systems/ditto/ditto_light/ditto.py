import logging
import os
from copy import copy
import random

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup

from graph_aug_for_er.augmenter.operator_mixup import apply_mixup
from graph_aug_for_er.graph_augmentation.graph_augmenter import GraphAugmentation
from graph_aug_for_er.utils.early_stopping import EarlyStopper, StopTrainingWhenTrainLossIsNearZero
from .dataset import DittoDataset

# from apex import amp

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}


class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8, experiment=None):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug
        self.experiment = experiment

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1, x2=None, labels=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device)  # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device)  # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size]  # (batch_size, emb_size)
            enc2 = enc[batch_size:]  # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            embeddings = self.bert.embeddings(input_ids=x1)

            if "mixup" in self.experiment.online_augmentation and labels is not None:
                left, right, labels = apply_mixup(embeddings, labels)

            head_mask = [None] * self.bert.config.num_hidden_layers
            attention_mask = torch.ones((len(embeddings), len(embeddings[0]))).to(self.device)

            enc = self.bert.transformer(
                x=embeddings,
                attn_mask=attention_mask,
                head_mask=head_mask,
                return_dict=True
            )
            enc = enc[0][:, 0, :]
            # enc = self.bert(x1)[0][:, 0, :]

        if labels is not None:
            return self.fc(enc), labels
        return self.fc(enc)


def evaluate(model, iterator, threshold=None, return_probs=False):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()

            all_y += y.cpu().long().numpy().tolist()

    if return_probs:
        return all_probs

    # here we need all y as integer labels for F1 score


    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0  # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(graph_augmentation: GraphAugmentation, model, optimizer, scheduler, hp, valid_iter, best_dev_f1, epoch,
               experiment, device):
    """Perform a single training step

    Args:
        df_train (pd.DataFrame)
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    avg_training_loss = 0.0

    ################### hard examples training phase #####################
    list_with_hardness = []
    with torch.no_grad():
        logging.info("feeding forward all pairs to determine the loss")
        for batch in tqdm(
                graph_augmentation.all_pairs_dataloader(),
                total=int(len(graph_augmentation.all_pairs) / experiment.batch_size),
        ):
            input_ids = []
            labels = []

            for left_index, right_index, label in batch:
                current_input_ids = graph_augmentation.construct_single_data_point(
                    left_index, right_index
                )
                input_ids.append(current_input_ids)
                labels.append(label)

            input_ids = torch.stack(input_ids)
            logits = model(input_ids)

            ce_loss = F.cross_entropy(logits, torch.tensor(labels).long().to(device), reduction="none")

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
    logging.info(f"Removed {len(list_with_hardness) - len(list_with_hardness_cleaned)} of {len(list_with_hardness)} entries")

    list_with_hardness_cleaned = sorted(list_with_hardness_cleaned, key=lambda x: x[3], reverse=True)
    positives_list = [i for i in list_with_hardness_cleaned if i[2] == 1]
    negatives_list = [i for i in list_with_hardness_cleaned if i[2] == 0]

    # try this: remove the hardest 1% of the lists as the probability for label noise is highest
    positives_list = positives_list[20:]

    negatives_list = sorted(negatives_list, key=lambda x: x[3], reverse=True)
    negatives_list = negatives_list[20:]

    final_list = copy(positives_list)

    # select amount to train on todo: do not hardcode "label". parametrize
    df_train = graph_augmentation.df_train
    pos_neg_ratio = df_train[df_train.label == 1].shape[0] / df_train[df_train.label == 0].shape[0]

    negatives_list = sorted(negatives_list, key=lambda x: x[3], reverse=True)

    expand_factor = min([int(1 / pos_neg_ratio), graph_augmentation.pos_neg_ratio_cap])

    final_list.extend(negatives_list[: len(positives_list) * expand_factor])

    random.shuffle(final_list)

    for i, batch in enumerate(graph_augmentation.batch_iterable(final_list)):
        optimizer.zero_grad()

        input_ids = []
        labels = []

        for left_index, right_index, label, calculated_loss in batch:
            left = graph_augmentation.get_datapoint_from_table_idx(left_index)
            right = graph_augmentation.get_datapoint_from_table_idx(right_index)

            current_input_ids = graph_augmentation.tokenize(left, right)
            input_ids.append(current_input_ids)

            labels.append(label)

        input_ids = torch.stack(input_ids)
        y = torch.tensor(labels).long().to(device)

        prediction = model(input_ids)

        loss = criterion(prediction, y)
        loss.backward()

        optimizer.step()
        scheduler.step()
        if i % 40 == 0:  # monitoring
            print(f"step: {i} / {int(len(final_list) / experiment.batch_size)}, loss: {loss.item()}")

        running_loss += loss.item()
        del loss

        # validate every n steps
        if i % 10000 == 0 and not i == 0:
            model.eval()
            valid_f1, _ = evaluate(model, valid_iter)
            print(f"inter-epoch validation. F1: {valid_f1}")
            if valid_f1 > best_dev_f1:
                best_dev_f1 = valid_f1
                if hp.save_model:
                    # create the directory if not exist
                    directory = os.path.join(hp.logdir, hp.task)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # save the checkpoints for each component
                    ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                    ckpt = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch}
                    torch.save(ckpt, ckpt_path)
            model.train()

    avg_training_loss = running_loss / i

    return best_dev_f1, avg_training_loss


def train(df_train: pd.DataFrame, trainset, validset, testset, run_tag, hp, experiment):
    """Train and evaluate the model

    Args:
        df_train (pd.DataFrame): the training set
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = validset.pad
    # # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size * 16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                batch_size=hp.batch_size * 16,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug,
                       experiment=experiment)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    # if hp.fp16:
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (
                            len(df_train) * 5 // hp.batch_size) * hp.n_epochs  # factor 5 is arbitrary, but we dont want to end with a learning rate of 0 because we created a lot more data with aug.
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)

    best_dev_f1 = best_test_f1 = 0.0
    epochs_trained = 0
    avg_train_loss = 0.0

    early_stopper = EarlyStopper()
    ckpt = {
        "model": model.state_dict()}  # saved so we have that "model" key and are able access even if we end up not saving the model here due to intra-epoch validation

    # graph augmentation instanciation here:
    if experiment.dataset == "wdc":
        cols_left = ["title_left", "brand_left", "category_left", "description_left"]
        cols_right = ["title_right", "brand_right", "category_right", "description_right"]
    elif experiment.dataset == "markt_pilot":
        cols_left = ["searched_brand", "searched_number", "searched_name", "searched_group", "searched_description"]
        cols_right = ["found_brand", "found_number", "found_name", "found_group", "found_description"]
    elif experiment.dataset in ["amazon_google", "walmart_amazon"]:
        cols = df_train.columns.tolist()
        cols_left = [i for i in cols if "left" in i]
        cols_right = [i for i in cols if "right" in i]
        df_train[cols_left] = df_train[cols_left].astype(str)
        df_train[cols_right] = df_train[cols_right].astype(str)
    else:
        raise NotImplementedError

    graph_augmentation = GraphAugmentation(
        df_train=df_train, cols_left=cols_left, cols_right=cols_right, label="label", experiment=experiment
    )

    logging.info(f"======== Initial Training Phase ========")
    criterion = nn.CrossEntropyLoss()
    model.train()
    for i, batch in enumerate(graph_augmentation.all_pairs_dataloader_for_initial_training()):
        optimizer.zero_grad()

        input_ids = batch["input_ids"]
        y = batch["labels"]

        prediction = model(input_ids)

        loss = criterion(prediction, y.long().to(model.device))
        loss.backward()

        optimizer.step()
        if i % 40 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")

        del loss

    logging.info(f"======== Starting hard examples training phase ========")
    for epoch in range(1, hp.n_epochs + 1):
        # train
        model.train()

        if avg_train_loss == 0.0 or not StopTrainingWhenTrainLossIsNearZero.training_loss_is_near_zero(avg_train_loss):
            best_dev_f1, avg_train_loss = train_step(graph_augmentation, model, optimizer, scheduler, hp, valid_iter, best_dev_f1,
                                                     epoch, experiment, device)

            # eval
            model.eval()
            dev_f1, th = evaluate(model, valid_iter)
            test_f1 = evaluate(model, test_iter, threshold=th)

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                if hp.save_model:
                    # create the directory if not exist
                    directory = os.path.join(hp.logdir, hp.task)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # save the checkpoints for each component
                    ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                    ckpt['model'] = model.state_dict()
                    ckpt['optimizer'] = optimizer.state_dict()
                    ckpt['scheduler'] = scheduler.state_dict()
                    ckpt['epoch'] = epoch
                    torch.save(ckpt, ckpt_path)

            print(f"epoch {epoch}: val_f1={dev_f1}, test_f1={test_f1}, best_test_f1={best_test_f1}")

            # logging
            scalars = {'f1': dev_f1,
                       't_f1': test_f1}
            writer.add_scalars(run_tag, scalars, epoch)

            epochs_trained = epoch
            if early_stopper.early_stop(1 - dev_f1):
                break

    writer.close()

    # evaluate with best model on test set after training is finished
    model.load_state_dict(ckpt["model"])
    all_probs = evaluate(model, test_iter, return_probs=True)
    return all_probs, epochs_trained
