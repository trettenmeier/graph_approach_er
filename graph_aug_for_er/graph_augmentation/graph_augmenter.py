import logging
import random
from typing import List, Union, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from graph_aug_for_er.utils.ditto import clean_col_name
from graph_aug_for_er.utils.load_config import ExperimentConfiguration


class GraphAugmentation:
    def __init__(self, df_train: pd.DataFrame, cols_left: list, cols_right: list, label: str,
                 experiment: ExperimentConfiguration):
        self.df_train = df_train
        self.cols_left = cols_left
        self.cols_right = cols_right
        self.label = label
        self.experiment = experiment

        self.label_noise_min_degree = experiment.label_noise_min_degree
        self.label_noise_threshold = experiment.label_noise_threshold
        self.pos_neg_ratio_cap = experiment.pos_neg_ratio_cap

        self.original_dataset_structure = []
        self.label_noise_set = set()

        self.G = nx.Graph()
        self.pos_graph, self.neg_graph, self.data, self.all_nodes = self.create_graph(df_train)

        self.all_pairs = self.train_generate_all_pairs()

        if self.experiment.model == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif self.experiment.model == "ditto":
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        else:
            raise NotImplementedError("Unkown model type")

    def create_graph(self, df):
        pos_graph = nx.Graph()
        neg_graph = nx.Graph()

        table = {}
        pos_edges = []
        neg_edges = []
        all_edges = []
        all_nodes = []

        logging.info("Setting up the graph")
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            if self.experiment.model == "bert":
                left = row[self.cols_left].values.tolist()
                right = row[self.cols_right].values.tolist()
                left = " ".join([str(i) for i in left if i is not None and i != np.nan and i.lower() != "null"])
                right = " ".join([str(i) for i in right if i is not None and i != np.nan and i.lower() != "null"])
            elif self.experiment.model == "ditto":
                left = right = ""
                for col in self.cols_left:
                    if row[col] is None or row[col] == np.nan or row[col] == "null":
                        continue
                    left = left + f"COL {clean_col_name(col)} VAL {row[col]} "

                for col in self.cols_right:
                    if row[col] is None or row[col] == np.nan or row[col] == "null":
                        continue
                    right = right + f"COL {clean_col_name(col)} VAL {row[col]} "

                left = left.strip()
                right = right.strip()

            else:
                raise NotImplementedError("unknown model type")

            hashed_left = hash(left)
            hashed_right = hash(right)

            self.G.add_node(hashed_left)
            self.G.add_node(hashed_right)
            all_edges.append((hashed_left, hashed_right))

            pos_graph.add_node(hashed_left)
            pos_graph.add_node(hashed_right)
            neg_graph.add_node(hashed_left)
            neg_graph.add_node(hashed_right)
            all_nodes.append(hashed_left)
            all_nodes.append(hashed_right)

            self.original_dataset_structure.append(
                (hashed_left, hashed_right, row.label)
            )

            if row.label == 1:
                pos_edges.append((hashed_left, hashed_right))
            else:
                neg_edges.append((hashed_left, hashed_right))

            table[hashed_left] = left
            table[hashed_right] = right

        pos_graph.add_edges_from(pos_edges)
        neg_graph.add_edges_from(neg_edges)
        self.G.add_edges_from(all_edges)

        return pos_graph, neg_graph, table, list(set(all_nodes))

    def get_positive(self, node_name: int) -> Union[List[int], None]:
        """
        Input: The hash-value of the current node. We are looking for all known positive matches
        """
        pos_connected = list(nx.node_connected_component(self.pos_graph, node_name))

        if len(pos_connected) == 1:  # if the node is not connected we can't find positives for that node
            return None

        random.shuffle(pos_connected)
        return pos_connected

    def get_negative(self, node_name: int) -> Union[List[int], None]:
        """
        Input: The hash value of the current node. We are looking for all known negative matches
        """
        pos_connected = list(nx.node_connected_component(self.pos_graph, node_name))

        # we look for all negatively connected nodes for the nodes in pos_connected. we can only use the neighbors for that
        # and not the subgraph as for a neg. connection: For A-B-C (negatively connected) we don't know anything about the
        # relationship of A-C
        neg_connected = []
        for node in pos_connected:
            adj_nodes = list(self.neg_graph.neighbors(node))
            neg_connected.extend(adj_nodes)
        neg_connected = list(set(neg_connected))

        # we can also use all positive connected nodes of these
        extended_connected = []
        for node in neg_connected:
            temp_list = list(nx.node_connected_component(self.pos_graph, node))
            extended_connected.extend(temp_list)

        # put everything together: this is the list of candidates we can choose neg. samples from
        extended_connected.extend(neg_connected)
        neg_connected = list(set(neg_connected))

        neg_connected = [i for i in neg_connected if i != node_name]

        random.shuffle(neg_connected)

        if len(neg_connected) == 0:
            return None

        return neg_connected

    def get_datapoint_from_table_idx(self, idx):
        item = self.data[idx]
        return item[:1000]  # cap length

    def tokenize(self, left: str, right: str) -> Tuple:
        encoded_dict = self.tokenizer(
            text=left,
            text_pair=right,
            add_special_tokens=True,
            return_attention_mask=False,
            return_tensors="pt",
            max_length=self.experiment.max_input_length,
            padding="max_length",
            truncation=True,
        )
        if self.experiment.model == "bert":
            return encoded_dict["input_ids"].squeeze(), encoded_dict["token_type_ids"].squeeze()
        else:
            return encoded_dict["input_ids"].squeeze()

    def train_generate_all_pairs(self) -> list:
        """
        This method generates all possible pairs and filters out duplicates. It also tries to detect inconsistencies that
        indicate label noise and removes them.
        """
        all_pairs = []
        logging.info("Generating all pairs")
        for node in tqdm(self.all_nodes):
            positives = self.get_positive(node)
            negatives = self.get_negative(node)

            if positives:
                all_pairs.extend([(node, i, 1) for i in positives if node != i])
            if negatives:
                all_pairs.extend([(node, i, 0) for i in negatives if node != i])

        positive = len([i for i in all_pairs if i[2] == 1])
        negative = len([i for i in all_pairs if i[2] == 0])
        logging.info(f"Generated {len(all_pairs)} pairs ({positive} positive and {negative} negative)")

        return self.deduplicate_all_pairs(all_pairs)

    def deduplicate_all_pairs(self, all_pairs: list) -> list:
        """
        deduplicate but also considering order switching
        """
        seen = set()
        seen_with_label = set()
        clean_list = []

        logging.info("Deduplicating all pairs list")
        for pair in tqdm(all_pairs, miniters=5000):
            temp_set = frozenset({pair[0], pair[1]})
            temp_set_with_label = frozenset({pair[0], pair[1], pair[2]})

            if temp_set in seen:
                if pair[2] == 0 and {pair[0], pair[1], 1} in seen_with_label:
                    self.label_noise_set.add(pair[0])
                    self.label_noise_set.add(pair[1])

                if pair[2] == 1 and {pair[0], pair[1], 0} in seen_with_label:
                    self.label_noise_set.add(pair[0])
                    self.label_noise_set.add(pair[1])
                continue
            else:
                seen.add(temp_set)
                clean_list.append(pair)
                seen_with_label.add(temp_set_with_label)

        # stats
        positive = len([i for i in clean_list if i[2] == 1])
        negative = len([i for i in clean_list if i[2] == 0])
        logging.info(f"After deduplicating: {len(clean_list)} pairs ({positive} positive and {negative} negative)")

        # remove identified label noise
        clean_list = [i for i in clean_list if i[0] not in self.label_noise_set and i[1] not in self.label_noise_set]
        logging.info(
            f"Label noise removal: Removed {len(self.label_noise_set)} data points. Final data points: {len(clean_list)}")

        return clean_list

    def all_pairs_dataloader_for_initial_training(self):
        batch_size = self.experiment.batch_size

        # remove label noise as detected by consistency check
        dataset = [i for i in self.original_dataset_structure if
                   i[0] not in self.label_noise_set and i[1] not in self.label_noise_set]
        logging.info(
            f"Removed {len(self.original_dataset_structure) - len(dataset)} data points. {len(dataset)} datapoints to train on.")

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i: i + batch_size]
            if self.experiment.model == "bert":
                input_ids, token_type_ids, labels = self.construct_data_points(batch)
                yield {"input_ids": input_ids, "token_type_ids": token_type_ids, "labels": labels}
            else:
                input_ids, labels = self.construct_data_points_for_ditto(batch)
                yield {"input_ids": input_ids, "labels": labels}

    def all_pairs_dataloader(self):
        batch_size = self.experiment.batch_size

        for i in range(0, len(self.all_pairs), batch_size):
            yield self.all_pairs[i: i + batch_size]

    def construct_data_points(self, batch):
        input_ids = []
        token_type_ids = []
        labels = []

        for left_index, right_index, label in batch:
            current_input_ids, current_token_type_ids = self.construct_single_data_point(left_index, right_index)
            input_ids.append(current_input_ids)
            token_type_ids.append(current_token_type_ids)
            labels.append(label)

        input_ids = torch.stack(input_ids)
        token_type_ids = torch.stack(token_type_ids)
        labels = torch.tensor(labels)

        return input_ids, token_type_ids, labels

    def construct_data_points_for_ditto(self, batch):
        input_ids = []
        labels = []

        for left_index, right_index, label in batch:
            current_input_ids = self.construct_single_data_point(left_index, right_index)
            input_ids.append(current_input_ids)
            labels.append(label)

        input_ids = torch.stack(input_ids)
        labels = torch.tensor(labels)

        return input_ids, labels

    def construct_single_data_point(self, left_index, right_index):
        if self.experiment.model == "bert":
            left = self.get_datapoint_from_table_idx(left_index)
            right = self.get_datapoint_from_table_idx(right_index)
            input_ids, token_type_ids = self.tokenize(left, right)
            return input_ids, token_type_ids
        if self.experiment.model == "ditto":
            left = self.get_datapoint_from_table_idx(left_index)
            right = self.get_datapoint_from_table_idx(right_index)
            input_ids = self.tokenize(left, right)
            return input_ids

    def batch_iterable(self, iterable):
        batch_size = self.experiment.batch_size

        for i in range(0, len(iterable), batch_size):
            yield iterable[i: i + batch_size]

    @staticmethod
    def find_label_noise(list_with_hardness: list, threshold: int, min_degree_to_consider: int) -> set:
        G = nx.Graph()
        all_nodes_to_add = [(i[0], i[1], i[3]) for i in list_with_hardness]
        G.add_weighted_edges_from(all_nodes_to_add)

        # Calculate the average weight of all edges in the graph
        total_weight = sum(w["weight"] for u, v, w in G.edges(data=True))
        avg_weight = total_weight / G.number_of_edges()

        # Identify nodes with degrees larger than 2
        nodes_over_min_degree = [node for node, degree in dict(G.degree()).items() if degree >= min_degree_to_consider]

        # Calculate the average weight of edges incident to each node
        significant_nodes = []
        for node in nodes_over_min_degree:
            incident_edges = G.edges(node, data=True)
            total_weight_incident = sum(w["weight"] for u, v, w in incident_edges)
            avg_weight_incident = total_weight_incident / len(incident_edges)
            if avg_weight_incident > avg_weight * threshold:
                significant_nodes.append(node)

        return set(significant_nodes)
