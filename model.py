import torch.nn as nn
from pytorch_pretrained_bert import BertModel

from defaults import BERT_MODEL


class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(
            tokens, attention_mask=masks, output_all_encoded_layers=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.sigmoid(linear_output)
        return proba
