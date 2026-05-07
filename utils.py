import os
import pandas as pd
from transformers import BertTokenizerFast
from MyDataset import ToutiaoDataset
import torch

import json
import numpy as np
label2id, id2label=None, None
def get_next(prefix_dir):
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir+'/exp1')
        return prefix_dir+'/exp1'
    else:
        existing_nums = []
        for file in os.listdir(prefix_dir):
            if file.startswith('exp'):
                existing_nums.append(int(file[3:]))
        if len(existing_nums) == 0:
            next_num = 1
        else:        
            next_num = max(existing_nums) + 1
        os.makedirs(prefix_dir+'/exp'+str(next_num))
        return prefix_dir+'/exp'+str(next_num)

def build_label_mappings(labels, save_path=None):
    unique_labels = set()
    for seq in labels:
        for tag in seq:
            unique_labels.add(tag)
    unique_labels = sorted(unique_labels)
    
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    if save_path:
        mappings = {"label2id": label2id, "id2label": id2label}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(mappings, f, ensure_ascii=False, indent=4)
        print(f"Label mappings saved to {save_path}")
    
    return label2id, id2label

def get_sentences(dir_path):
    sentence = []
    tag = []
    sentences_list = []
    tags_list = []
    for line in open(dir_path,encoding='utf-8'):
        if line[0] == '\n':
            
            sentences_list.append(sentence)
            tags_list.append(tag)
            sentence = []
            tag = []
            continue
        else:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            sentence.append(parts[0])
            tag.append(parts[1])
    return {'sentences':sentences_list,'tags':tags_list}

        

        
    
def load_data(config):
    data_dir=config.data_path
    headers=[
    "id",
    "label",
    "channel",
    "title",
    "keywords"
]
    train_data = get_sentences(os.path.join(data_dir, 'train.txt'))
    test_data = get_sentences(os.path.join(data_dir, 'test.txt'))
    dev_data = get_sentences(os.path.join(data_dir, 'dev.txt'))
    label2id, id2label = build_label_mappings(train_data['tags'], save_path=os.path.join(data_dir, 'label2id.json'))
    config.set_class_num(len(label2id))
    known_labels = set(label2id.keys())
    tags_list = {
        "train": train_data["tags"],
        "dev": dev_data["tags"],
        "test": test_data["tags"]
    }
    for name, tags_list in tags_list.items():
        unknown_labels = set()
        for i in range(len(tags_list)):
            for j in range(len(tags_list[i])):
                if tags_list[i][j] not in known_labels:
                    unknown_labels.add(tags_list[i][j])
                else:
                    tags_list[i][j] = label2id[tags_list[i][j]]
        
        if unknown_labels:
            raise ValueError(
                f"{name} 集中存在训练集 label2id 未覆盖的标签: {sorted(unknown_labels)}"
            )
    tokenizer = BertTokenizerFast.from_pretrained(config.model_dir)
    train_dataset = ToutiaoDataset(train_data,tokenizer,config.max_length)
    test_dataset = ToutiaoDataset(test_data,tokenizer,config.max_length)
    dev_dataset = ToutiaoDataset(dev_data,tokenizer,config.max_length)

    train_dataLoader = train_dataset.get_data_loader(batch_size=config.batch_size)
    dev_dataLoader = dev_dataset.get_data_loader(batch_size=config.batch_size,shuffle=False)
    test_dataLoader = test_dataset.get_data_loader(batch_size=config.batch_size,shuffle=False)
    return train_dataLoader, dev_dataLoader, test_dataLoader

def write_log(log_jsonl_path, log_dict):

    with open(log_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
class Metrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        print(f"num_classes: {num_classes}")
        self.confusion_matrix = [[0 for _ in range(self.num_classes)] 
                                  for _ in range(self.num_classes)]
        self.result_df = None
        
    def add(self, predictions, labels):    
        predictions = predictions.tolist()
        labels = labels.tolist()

        for pred, true in zip(predictions, labels):
            if true == -100:
                continue
            self.confusion_matrix[true][pred] += 1
    def reset(self):
        self.confusion_matrix = [[0 for _ in range(self.num_classes)] 
                                  for _ in range(self.num_classes)]
        self.result_df = None
    
    def calculate_tp_fp_fn(self, class_id): 
        tp = self.confusion_matrix[class_id][class_id]
        fp = sum(self.confusion_matrix[i][class_id] for i in range(self.num_classes)) - tp
        fn = sum(self.confusion_matrix[class_id]) - tp
        return tp, fp, fn
    
    def precision(self, class_id=None):
        if class_id is not None:
            tp, fp, _ = self.calculate_tp_fp_fn(class_id)
            if tp + fp == 0:
                return 0.0
            return tp / (tp + fp)
        else :
            total_tp,total_fp,total_fn = 0,0,0
            for i in range(self.num_classes):
                tp, fp, fn = self.calculate_tp_fp_fn(i)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            if total_tp + total_fp == 0:
                return 0.0
            return total_tp /(total_tp + total_fp)
    def get_result_dict(self):
        df = self.result_df
        result = {}

        for idx in df.index:
            if isinstance(idx, int):
                key = f"class_{idx}"
            else:
                key = str(idx)  
            
            row = df.loc[idx]
            row_dict = {}
            for col in df.columns:
                val = row[col]
               
                if pd.isna(val):
                    row_dict[col] = None
                
                elif isinstance(val, (np.integer, np.floating)):
                    row_dict[col] = val.item()
                else:
                    row_dict[col] = val
            result[key] = row_dict

        return result

    
    def recall(self, class_id=None):
       
        if class_id is not None:
            tp, fp, fn = self.calculate_tp_fp_fn(class_id)
            if tp + fn == 0:
                return 0.0
            return tp / (tp + fn)
        else:
            total_tp,total_fp,total_fn = 0,0,0
            for i in range(self.num_classes):
                tp, fp, fn = self.calculate_tp_fp_fn(i)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            if total_tp + total_fn == 0:
                return 0.0
            return total_tp /(total_tp + total_fn)
    
    def f1_score(self, class_id=None):
       
        if class_id is not None:
            p = self.precision(class_id)
            r = self.recall(class_id)
            if p + r == 0:
                return 0.0
            return 2 * p * r / (p + r)
        
        else:
            total_tp,total_fp,total_fn = 0,0,0
            for i in range(self.num_classes):
                tp, fp, fn = self.calculate_tp_fp_fn(i)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            if total_tp + total_fn == 0:
                r = 0.0
            else:
                r = total_tp /(total_tp + total_fn)
            if total_tp + total_fp == 0:
                p = 0.0
            else:
                p = total_tp /(total_tp + total_fp)
            if p + r == 0:
                return 0.0
            return 2 * p * r / (p + r)
    def get_results(self):
        p_list=[]
        r_list=[]
        f1_list=[]
        support_list = []

        for i in range(self.num_classes):
            p_list.append(self.precision(i))
            r_list.append(self.recall(i))
            f1_list.append(self.f1_score(i))
            support=sum(self.confusion_matrix[i])
            support_list.append(support)
        micro_p = self.precision()      
        micro_r = self.recall()         
        micro_f1 = self.f1_score() 
        df=pd.DataFrame({'precision':p_list,'recall':r_list,'f1_score':f1_list,'support':support_list})
        df.loc['macro_avg'] = df[['precision', 'recall', 'f1_score']].mean()
        df.loc['macro_avg', 'support'] = float('nan')
        df.loc['micro_avg'] = [micro_p,micro_r,micro_f1,float('nan')]
        
        self.result_df = df
        return df
class EarlyStop():
    def __init__(self,config,save_dir=None):
        self.config=config
        self.monitor = config.monitor
        self.delta=config.delta
        self.best_score = None
        self.counter = 0
        self.patience = config.patience
        self.early_stop = False
        self.save_dir = save_dir
        
    def __call__(self, epoch,loss,acc,f1_score, model,optimizer,scheduler,):
        if self.monitor == 'val_acc':
            if self.best_score is None :
                self.best_score = acc
                
                self.save_checkpoint(model, optimizer, scheduler, epoch,acc,True)
                return 
            

            if acc-self.best_score  < self.delta:
                self.counter += 1
                self.save_checkpoint(model, optimizer, scheduler, epoch,acc,False)
                if self.counter > self.patience:
                    self.early_stop = True
            else:
                self.best_score = acc
                self.counter = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch,acc,True)
        elif self.monitor == 'val_loss':
            if self.best_score is None:
                self.best_score = loss
                self.save_checkpoint(model, optimizer, scheduler, epoch,loss,True)
                return
            if self.best_score - loss  < self.delta:
                self.counter += 1
                self.save_checkpoint(model, optimizer, scheduler, epoch,loss,False)
                if self.counter > self.patience:
                    self.early_stop = True
            else:
                self.best_score = loss
                self.counter = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch,loss,True)
        if self.monitor == 'val_f1':
            if self.best_score is None :
                self.best_score = f1_score
                
                self.save_checkpoint(model, optimizer, scheduler, epoch,f1_score,True)
                return 
            

            if f1_score-self.best_score  < self.delta:
                self.counter += 1
                self.save_checkpoint(model, optimizer, scheduler, epoch,f1_score,False)
                if self.counter > self.patience:
                    self.early_stop = True
            else:
                self.best_score = f1_score
                self.counter = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch,f1_score,True)
        return self.early_stop
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, dev_metrics,is_best):
        checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        checkpoint = {
            'epoch': epoch,
            'label2id': label2id,
            'id2label': id2label,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"保存 epoch {epoch + 1} 的 checkpoint: {checkpoint_path}")
        if is_best:
            self.best_model_path = checkpoint_path
            print(f"更新最佳模型: {checkpoint_path} (监控指标 '{self.monitor}' = {dev_metrics:.6f})")
class Arguments:
    def __init__(self, config_path="arguments.json"):
        self.args_dict = self._load_json_config(config_path)
        self.class_num=None
        for key, value in self.args_dict.items():
            setattr(self, key, value)
        
    def _load_json_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    def get_args_dict(self):
        return self.args_dict
    def set_class_num(self,class_num):
        self.class_num=class_num
        self.args_dict['class_num']=class_num
        

if __name__ == '__main__': 
    args = Arguments("args/arg1.json")

    train_dataloader, dev_dataloader, test_dataloader = load_data(args)
    print(args.get_args_dict())
    for input_ids, attention_mask, targets in train_dataloader:
        print(input_ids[0])
        print(targets[0])
        break
    
        
   
