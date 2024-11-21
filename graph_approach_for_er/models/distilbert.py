import torch

from transformers import DistilBertForSequenceClassification

from graph_approach_for_er.utils.load_config import ExperimentConfiguration


def get_model(experiment: ExperimentConfiguration):
    return Model(experiment=experiment)


class Model(torch.nn.Module):
    def __init__(self, experiment: ExperimentConfiguration):
        super(Model, self).__init__()
        self.experiment = experiment

        self.bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).long()
        output = self.bert(input_ids, attention_mask=attention_mask)
        return {"logits": output["logits"], "last_hidden_state": None}
