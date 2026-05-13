import os
import pandas as pd
from transformers import BertTokenizerFast
from MyDataset import WeiboNerDataset
import torch

import json
import numpy as np
global label2id, id2label
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
    train_data = get_sentences(os.path.join(data_dir, 'train.txt'))
    test_data = get_sentences(os.path.join(data_dir, 'test.txt'))
    dev_data = get_sentences(os.path.join(data_dir, 'dev.txt'))
    label2id, id2label = build_label_mappings(train_data['tags']+test_data['tags']+dev_data['tags'], save_path=os.path.join(data_dir, 'label2id.json'))
    config.set_mapping(label2id,id2label)
    tokenizer = BertTokenizerFast.from_pretrained(config.model_dir)
    train_dataset = WeiboNerDataset(train_data, tokenizer, config.max_length, label2id, config.align_type)
    test_dataset = WeiboNerDataset(test_data, tokenizer, config.max_length, label2id, config.align_type)
    dev_dataset = WeiboNerDataset(dev_data, tokenizer, config.max_length, label2id, config.align_type)

    train_dataLoader = train_dataset.get_data_loader(batch_size=config.batch_size)
    dev_dataLoader = dev_dataset.get_data_loader(batch_size=config.batch_size,shuffle=False)
    test_dataLoader = test_dataset.get_data_loader(batch_size=config.batch_size,shuffle=False)
    return train_dataLoader, dev_dataLoader, test_dataLoader,label2id,id2label

def write_log(log_jsonl_path, log_dict):

    with open(log_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
class Metrics:
    def __init__(self,label2id,id2label):
        self.id2label = id2label
        self.label2id = label2id
        self.entity_types = set()
        for idx, label in id2label.items():
            if label.startswith('B-'):
                self.entity_types.add(label[2:])  
        self.entity_types = sorted(self.entity_types)
        self.all_true_entities = []
        self.all_pred_entities = []
        self._counts = None
        
    def add(self, predictions, labels):    
        predictions = predictions.tolist()
        labels = labels.tolist()
        for pred_seq, label_seq in zip(predictions, labels):
        
            pred_str = [self.id2label.get(p, 'O') if p != -100 else 'O' for p in pred_seq]
            true_str = [self.id2label.get(l, 'O') if l != -100 else 'O' for l in label_seq]
            self.all_true_entities.extend(self._extract_entities(true_str))
            self.all_pred_entities.extend(self._extract_entities(pred_str))
    def _extract_entities(self, tags):
        entities =[]
        i=0
        while i<len(tags):
            if tags[i].startswith('B-'):
                entity_type = tags[i][2:]
                start=i
                i+=1
                while i<len(tags) and tags[i]=='I-'+entity_type:
                    i+=1
                end=i
                entities.append((entity_type,start,end))
            else:
                i+=1
        return entities


    def reset(self):
        self.all_true_entities = []
        self.all_pred_entities = []
        self._counts = None
        self.confusion_matrix = [[0 for _ in range(len(self.entity_types))] 
                                  for _ in range(len(self.entity_types))]
        self.result_df = None
    
    def _compute_counts(self):
        true_list = self.all_true_entities   
        pred_list = self.all_pred_entities
        counts = {etype: {'tp': 0, 'fp': 0, 'fn': 0} for etype in self.entity_types}
        matched_true = [False] * len(true_list)
        matched_pred = [False] * len(pred_list)
        for i, etype in enumerate(true_list):
            if matched_true[i]:
                continue
            true_type,true_start, true_end = etype[0],etype[1], etype[2]
            for j, etype_j in enumerate(pred_list):
                if matched_pred[j]:
                    continue
                pre_type,pre_strat,pre_end=etype_j[0],etype_j[1], etype_j[2]
                if etype_j == etype:
                    counts[etype[0]]['tp'] += 1
                    matched_true[i] = True
                    matched_pred[j] = True
                    break
        for i, true_ent in enumerate(true_list):
            if not matched_true[i]:
                etype = true_ent[0]
                counts[etype]['fn'] += 1
        for i, pred_ent in enumerate(pred_list):
            if not matched_pred[i]:
                
                counts[pred_ent[0]]['fp'] += 1
        self._counts = counts
        return counts
    def precision(self, entity_type=None):
       
        if self._counts is None:
            self._compute_counts()
        counts = self._counts
        if entity_type is not None:
            if entity_type not in counts:
                return 0.0
            tp = counts[entity_type]['tp']
            fp = counts[entity_type]['fp']
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        else:
            # 宏观平均（所有类别未加权平均）
            if not counts:
                return 0.0
            total_prec = sum(self.precision(et) for et in self.entity_types)
            return total_prec / len(self.entity_types)
    def recall(self, entity_type=None):
        
        if self._counts is None:
            self._compute_counts()
        counts = self._counts
        if entity_type is not None:
            if entity_type not in counts:
                return 0.0
            tp = counts[entity_type]['tp']
            fn = counts[entity_type]['fn']
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            if not counts:
                return 0.0
            total_rec = sum(self.recall(et) for et in self.entity_types)
            return total_rec / len(self.entity_types)
    
    def f1_score(self, entity_type=None):
        if entity_type is not None:
            p = self.precision(entity_type)
            r = self.recall(entity_type)
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        else:
    
            p = self.precision()
            r = self.recall()
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    def get_results(self):
       
        counts = self._compute_counts()
        
        p_list = []
        r_list = []
        f1_list = []
        support_list = []   # 真实实体数量（FN+TP）
        
        for etype in self.entity_types:
            p = self.precision(etype)
            r = self.recall(etype)
            f1 = self.f1_score(etype)
            support = counts[etype]['tp'] + counts[etype]['fn']
            p_list.append(p)
            r_list.append(r)
            f1_list.append(f1)
            support_list.append(support)
        
       
        df = pd.DataFrame({
            'precision': p_list,
            'recall': r_list,
            'f1_score': f1_list,
            'support': support_list
        }, index=self.entity_types)
        
       
        df.loc['macro_avg'] = df[['precision', 'recall', 'f1_score']].mean()
        df.loc['macro_avg', 'support'] = float('nan')
        
        
        total_tp = sum(v['tp'] for v in counts.values())
        total_fp = sum(v['fp'] for v in counts.values())
        total_fn = sum(v['fn'] for v in counts.values())
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
        df.loc['micro_avg'] = [micro_p, micro_r, micro_f1, float('nan')]
        
        self.result_df = df
        return df
    
    def get_result_dict(self):
        
        if not hasattr(self, 'result_df'):
            df = self.get_results()
        else:
            df = self.result_df
        
        result = {}
        for idx in df.index:
           
            if isinstance(idx, int) or isinstance(idx, str):
                key = str(idx)
            else:
                key = idx   
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
        self.best_model_path=None
        
    def __call__(self, epoch,loss,acc,f1_score, model,optimizer,scheduler,):
        

        if self.monitor == 'val_acc':
            if self.best_score is None :
                self.best_score = acc
                
                self.save_checkpoint(model, optimizer, scheduler, epoch,acc,True)
                return 
            

            if acc-self.best_score  < self.delta:
                self.counter += 1
                
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
                
                if self.counter > self.patience:
                    self.early_stop = True
            else:
                self.best_score = f1_score
                self.counter = 0
                self.save_checkpoint(model, optimizer, scheduler, epoch,f1_score,True)
        if epoch==self.config.num_epochs-1:
            self.save_checkpoint(model, optimizer, scheduler, epoch,acc,False)
            return
        return self.early_stop
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, dev_metrics,is_best):
        checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        checkpoint = {
            'epoch': epoch,
            'label2id': self.config.label2id,
            'id2label': self.config.id2label,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if is_best and self.best_model_path is not None:
            os.remove(self.best_model_path)
        torch.save(checkpoint, checkpoint_path)
        print(f"保存 epoch {epoch + 1} 的 checkpoint: {checkpoint_path}")
        if is_best:
            self.best_model_path = checkpoint_path
            print(f"更新最佳模型: {checkpoint_path} (监控指标 '{self.monitor}' = {dev_metrics:.6f})")
class Arguments:
    def __init__(self, config_path="arguments.json"):
        self.args_dict = self._load_json_config(config_path)
        self.class_num=None
        self.label2id=None
        self.id2label=None
        for key, value in self.args_dict.items():
            setattr(self, key, value)
        
    def _load_json_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    def get_args_dict(self):
        return self.args_dict
    def set_mapping(self,label2id,id2label):
        self.label2id=label2id
        self.id2label=id2label
        self.class_num=len(label2id)
        self.args_dict['class_num']=self.class_num
        self.args_dict['label2id']=label2id
        self.args_dict['id2label']=id2label
        

