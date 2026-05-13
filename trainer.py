from utils import Metrics
from utils import EarlyStop
import swanlab
from tqdm import tqdm
import torch
from model import Bert4NER
import argparse
from utils import load_data
import torch.nn as nn
from utils import get_next,write_log,Arguments,Metrics
import os
class Trainer:
    def __init__(self,config):
        self.optimizer=None
        self.scheduler=None
        self.device=config.device
        self.num_epochs=config.num_epochs
        self.config=config
        self.loss_fn = nn.CrossEntropyLoss()
        print(config.get_args_dict())
        self.metrics = Metrics(config.label2id,config.id2label)
        self.best_accuracy = 0.0
        self.save_dir = get_next(config.save_dir)
        self.early_stop = EarlyStop(config, self.save_dir)
        self.log_dir=os.path.join(self.save_dir,"log.jsonl")
        
    def train(self,traindataLoader, devdataLoader, testdataLoader, model,optimizer, scheduler):
        self.optimizer=optimizer
        self.model=model
        model.to(self.device)
        self.scheduler=scheduler
        swanlab.init(
            project="weibo_ner",  
            name="bert4ner",                
            config={
                "num_epochs": self.config.num_epochs,
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "model": "bert-base-chinese"
            }
        )
        write_log(self.log_dir, {"config": self.config.get_args_dict()})
        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            progress_bar = tqdm(traindataLoader, desc=f"Epoch {epoch + 1}/{self.num_epochs} [Train]", position=0, leave=True)
            for step, (input_ids, attention_mask, labels) in enumerate(progress_bar):
                
                self.optimizer.zero_grad()
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels=labels.view(-1)
                labels = labels.to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss=self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})
                if step % 50 == 0:
                    swanlab.log({
                        "train/loss_step": loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]['lr']
                    }, step=epoch * len(traindataLoader) + step)
            avg_train_loss = total_train_loss / len(traindataLoader)
            swanlab.log({
                "train/loss_epoch": avg_train_loss
            }, step=epoch)
            avg_eval_loss ,eval_accuracy,results_dict = self.eval(epoch, devdataLoader)
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "eval/loss": avg_eval_loss,
                "eval/accuracy": eval_accuracy,
                "eval/f1": results_dict['micro_avg']['f1_score'],
                "eval/results": results_dict
            }
            f1=results_dict['micro_avg']['f1_score']
            write_log(self.log_dir, log_dict)
            if self.scheduler is not None:
                self.scheduler.step(avg_eval_loss)
            if self.early_stop(epoch,avg_eval_loss,eval_accuracy,f1, model,optimizer,scheduler):
                break

        self.test(testdataLoader)
        swanlab.finish()
            
    
        
    def eval(self,epoch, devdataLoader):
        self.model.eval()
        
        total_eval_loss = 0
        eval_correct = 0
        total_samples = 0
        progress_bar = tqdm(devdataLoader, desc="Evaluation", position=0, leave=True)
        with torch.no_grad():
            for input_ids, attention_mask, labels in progress_bar:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels=labels.view(-1)
                labels = labels.to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss_fn = self.loss_fn
                loss = loss_fn(logits, labels)
                total_eval_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                eval_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                self.metrics.add(predictions, labels)
        results = self.metrics.get_results()
        print(results)
        results_dict = self.metrics.get_result_dict()
        self.metrics.reset()
        avg_eval_loss = total_eval_loss / len(devdataLoader)
        eval_accuracy = eval_correct / total_samples  
        print(f"Eval Accuracy: {eval_accuracy:.4f}")
        print(f"Eval Loss: {avg_eval_loss:.4f}")
        print(f"Eval F1 Score: {results_dict['micro_avg']['f1_score']:.4f}")
        swanlab.log({
            "eval/loss": avg_eval_loss,
            "eval/accuracy": eval_accuracy,
            "eval/f1": results_dict['micro_avg']['f1_score'],
            
        })

        
        return avg_eval_loss,eval_accuracy,results_dict
    def test(self, testdataLoader):
        checkpoint = torch.load(self.early_stop.best_model_path)
        self.model.load_state_dict(checkpoint["model"])  
        self.model = self.model.to(self.device)
        self.model.eval()
        total_test_loss = 0
        test_correct = 0
        total_samples = 0
        progress_bar = tqdm(testdataLoader, desc="Testing", position=0, leave=True)
        with torch.no_grad():
            for  input_ids, attention_mask, labels in progress_bar:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels=labels.view(-1)
                labels = labels.to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss_fn = self.loss_fn
                loss = loss_fn(logits, labels)
                total_test_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                test_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                self.metrics.add(predictions, labels)
        results = self.metrics.get_results()
        results_dict = self.metrics.get_result_dict()
        avg_test_loss = total_test_loss / len(testdataLoader)
        test_accuracy = test_correct / total_samples  
        print(f"Test Accuracy: {test_accuracy:.4f}")
        self.metrics.reset()
        log_dict = {
            "test/results": results_dict
        }
        write_log(self.log_dir, {"test": log_dict})
        swanlab.log({
            "test/loss": avg_test_loss,
            "test/accuracy": test_accuracy,
            
        })
        print(results_dict)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg', type=str, default='./args/arg1.json')
    args = parser.parse_args()
    args=Arguments(args.arg)
    traindataLoader, devdataLoader, testdataLoader,label2id,id2label = load_data(args)
    args.set_mapping(label2id,id2label)
    model=Bert4NER(args)
    optimizer, scheduler = model.get_optimizer()
    trainer=Trainer(args)
    trainer.train(traindataLoader, devdataLoader, testdataLoader, model, optimizer, scheduler)
   
        