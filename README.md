# BERT for Named Entity Recognition (NER)

基于 BERT 的中文命名实体识别（NER）项目，使用 PyTorch 和 Hugging Face Transformers 实现。

## 项目简介



完整标签列表（共 17 个标签）：

| 标签 | 标签 ID | 说明 |
|------|---------|------|
| B-GPE.NAM | 0 | 地缘政治实体-专有名词-开始 |
| B-GPE.NOM | 1 | 地缘政治实体-普通名词-开始 |
| B-LOC.NAM | 2 | 地点-专有名词-开始 |
| B-LOC.NOM | 3 | 地点-普通名词-开始 |
| B-ORG.NAM | 4 | 组织-专有名词-开始 |
| B-ORG.NOM | 5 | 组织-普通名词-开始 |
| B-PER.NAM | 6 | 人名-专有名词-开始 |
| B-PER.NOM | 7 | 人名-普通名词-开始 |
| I-GPE.NAM | 8 | 地缘政治实体-专有名词-内部 |
| I-GPE.NOM | 9 | 地缘政治实体-普通名词-内部 |
| I-LOC.NAM | 10 | 地点-专有名词-内部 |
| I-LOC.NOM | 11 | 地点-普通名词-内部 |
| I-ORG.NAM | 12 | 组织-专有名词-内部 |
| I-ORG.NOM | 13 | 组织-普通名词-内部 |
| I-PER.NAM | 14 | 人名-专有名词-内部 |
| I-PER.NOM | 15 | 人名-普通名词-内部 |
| O | 16 | 非实体 |

## 项目结构

```
bert-ner/
├── args/                    # 配置文件
│   ├── arg1.json           # 实验配置 1
│   └── arg2.json           # 实验配置 2
├── checkpoint/             # 模型检查点
│   └── exp*/              # 实验记录
│       └── log.jsonl      # 训练日志
├── data/                   # 数据集
│   ├── train.txt          # 训练集
│   ├── dev.txt            # 验证集
│   ├── test.txt           # 测试集
│   ├── label2id.json      # 标签映射
│   └── class.txt          # 类别信息
├── MyDataset.py           # 数据集加载器
├── model.py               # 模型定义
├── trainer.py             # 训练流程
├── utils.py               # 工具函数
├── requirements.txt       # 依赖包
└── README.md              # 项目文档
```

## 环境安装

```bash
pip install -r requirements.txt
```

### 核心依赖

- PyTorch >= 2.5.1
- Transformers >= 4.51.3
- SwanLab >= 0.7.5 (实验跟踪)
- Datasets >= 3.5.0
- Pandas >= 2.2.3
- NumPy >= 1.26.3

## 配置说明

配置文件位于 `args/arg1.json`：

```json
{
    "num_epochs": 100,        // 最大训练轮数
    "batch_size": 16,         // 批次大小
    "lr": 3e-5,              // 学习率
    "weight_decay": 0.01,    // 权重衰减
    "device": "cuda:0",      // 训练设备
    "model_dir": "../bert-base-chinese",  // BERT 模型路径
    "dropout_rate": 0.6,     // Dropout 比例
    "embedding_dim": 768,    // BERT 隐藏层维度
    "data_path": "./data/",  // 数据路径
    "max_length": 128,       // 最大序列长度
    "patience": 15,          // 早停耐心值
    "monitor": "val_f1",     // 监控指标
    "delta": 0.0001,         // 最小改进阈值
    "save_dir": "./checkpoint"  // 保存路径
}
```

## 使用方法

### 训练模型

```bash
python trainer.py
```


### 数据格式

数据文件采用 BIO 标注格式，每行一个词和标签，用空格分隔，句子之间用空行分隔：

```
北 B-LOC.NAM
京 I-LOC.NAM
是 O
中 B-GPE.NAM
国 I-GPE.NAM
的 O
首 O
都 O

```

## 实验结果

### 测试集性能 (exp16)

#### 总体指标

| 指标 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| **Macro Average** | 0.6668 | 0.5492 | **0.5788** |


#### 各类别详细结果

| 类别 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| **B-GPE.NAM** (地缘-专有) | 0.7627 | 0.9783 | 0.8571 | 46 |
| **B-GPE.NOM** (地缘-普通) | 0.0000 | 0.0000 | 0.0000 | 2 |
| **B-LOC.NAM** (地点-专有) | 0.8333 | 0.2632 | 0.4000 | 19 |
| **B-LOC.NOM** (地点-普通) | 0.6000 | 0.3333 | 0.4286 | 9 |
| **B-ORG.NAM** (组织-专有) | 0.5938 | 0.5000 | 0.5429 | 38 |
| **B-ORG.NOM** (组织-普通) | 1.0000 | 0.4375 | 0.6087 | 16 |
| **B-PER.NAM** (人名-专有) | 0.7213 | 0.8073 | 0.7619 | 109 |
| **B-PER.NOM** (人名-普通) | 0.7219 | 0.7485 | 0.7349| 163 |
| **I-GPE.NAM** (地缘-专有) | 0.8358 | 0.9333 | 0.8819 | 60 |
| **I-GPE.NOM** (地缘-普通) | 0.0000 | 0.0000 | 0.0000 | 2 |
| **I-LOC.NAM** (地点-专有) | 0.6875 | 0.4889 | 0.5714 | 45 |
| **I-LOC.NOM** (地点-普通) | 0.5000 | 0.2000 | 0.2857 | 15 |
| **I-ORG.NAM** (组织-专有) | 0.6176 | 0.6364 | 0.6269 | 99 |
| **I-ORG.NOM** (组织-普通) | 1.0000 | 0.4762 | 0.6452 | 21 |
| **I-PER.NAM** (人名-专有) | 0.7526 | 0.7259 | 7390 | 197 |
| **I-PER.NOM** (人名-普通) | 0.7222 | 0.8204 | 0.7682 | 206 |
| **O** (非实体) | 0.9871 | 0.9875 | 0.9873 | 13,485 |

## 模型架构

```
BERT-base-chinese
    ↓
Dropout (0.6)
    ↓
Linear (768 → 17)
    ↓
Softmax / CrossEntropyLoss
```





