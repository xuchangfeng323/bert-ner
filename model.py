from transformers import BertModel
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import Arguments
import torch.nn as nn

class Bert4TextClassification(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.lr=config.lr
        self.weight_decay=config.weight_decay
        self.bert = BertModel.from_pretrained(config.model_dir)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(config.embedding_dim, config.class_num)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=2, verbose=1, factor=0.1)
        return optimizer, lr_scheduler
