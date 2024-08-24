import torch

from transformers import BertModel

from graph_aug_for_er.utils.load_config import ExperimentConfiguration


def get_model(experiment: ExperimentConfiguration):
    return Model(experiment=experiment)


class Model(torch.nn.Module):
    def __init__(self, experiment: ExperimentConfiguration):
        super(Model, self).__init__()
        self.experiment = experiment

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(p=0.1)
        self.fc = torch.nn.Linear(in_features=self.bert.config.hidden_size, out_features=2)

    def forward(self, input_ids, token_type_ids):
        embeddings = self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        encoded = self.bert.encoder(embeddings)
        output = self.bert.pooler(encoded.last_hidden_state)

        output = self.dropout(output)

        return {"logits": self.fc(output), "last_hidden_state": encoded.last_hidden_state}

    def forward_from_last_hidden_state(self, last_hidden_state):
        output = self.bert.pooler(last_hidden_state)
        output = self.dropout(output)
        return self.fc(output)
