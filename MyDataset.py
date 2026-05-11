from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
import pandas as pd
class WeiboNerDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_length=128,label2id=None,align_type='ignore'):
        self.align_type=align_type
        self.texts = data['sentences']
        self.label_list = data['tags']
        self.label2id=label2id
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label_list[idx]
        return {
            'text': text,
            'labels': label
        }
    def collate_fn(self, batch):
        encodings = self.tokenizer([item['text'] for item in batch], 
                                   truncation=True, 
                                   is_split_into_words=True,
                                   padding='longest', 
                                   max_length=self.max_length, 
                                   return_tensors="pt")
        targets=[]
        labels=[item['labels'] for item in batch]
        for i,label in enumerate(labels):
            label_ids = []
            word_ids = encodings.word_ids(batch_index=i)
            current_word= None
            for word_idx in word_ids:
                if word_idx is None:
                    current_word=None
                    label_ids.append(-100)
                elif word_idx == current_word:
                    current_word = word_idx
                    if self.align_type == 'ignore':
                        label_ids.append(-100)
                    else:
                        label_ids.append(self.label2id[label[word_idx]])
                        current_word=word_idx
                else:
                    current_word=word_idx
                    label_ids.append(self.label2id[label[word_idx]])
            targets.append(label_ids)
        targets=torch.tensor(targets, dtype=torch.long)
        return encodings['input_ids'], encodings['attention_mask'], targets

            
    def get_data_loader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle)

    

