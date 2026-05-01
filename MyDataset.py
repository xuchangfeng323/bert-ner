from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
import pandas as pd

class ToutiaoDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_length=128):
        self.texts = data['sentences']
        self.labels = data['tags']
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return {
            'text': text,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    def collate_fn(self, batch):
        encodings = self.tokenizer([item['text'] for item in batch], 
                                   truncation=True, 
                                   is_split_into_words=True,
                                   padding='max_length', 
                                   max_length=self.max_length, 
                                   return_tensors="pt")
        targets=[]
        labels=[item['labels'] for item in batch]
        for i,label in enumerate(labels):
            label_ids = []
            word_ids = encodings.word_ids(batch_index=i) 
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_idx])
            targets.append(label_ids)
        targets=torch.tensor(targets, dtype=torch.long)
        return encodings['input_ids'], encodings['attention_mask'], targets

            
    def get_data_loader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

