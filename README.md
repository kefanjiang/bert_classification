# bert_classification
# 项目简介

本项目基于 [Amazon Polarity](https://huggingface.co/datasets/amazon_polarity) 数据集，使用 **BERT**（`bert-base-uncased`）模型进行二分类任务（正面/负面评论）。项目主要流程包括：

1. **加载并预处理数据**
2.  **加载bert模型** 
3. **训练与验证**  
4. **模型导出至 ONNX** 以实现快速推理  
5. **使用 ONNX Runtime 进行推理**  
6. **加载训练好的模型并在测试集上评估性能**

## 数据集说明

- **数据来源**: [Amazon Polarity 数据集](https://huggingface.co/datasets/amazon_polarity)  
- **数据规模**: 原始训练集 3,600,000 条，测试集 400,000 条；为了演示方便，代码中只使用了前 20,000 条训练数据和前 5,000 条测试数据，并在训练集上按 80:20 的比例拆分得出训练集与验证集。  
- **数据格式**:  
  - `content`: 评论文本  
  - `label`: 0 表示负面，1 表示正面

## 模型介绍

- **预训练模型**: [BertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification)  
- **模型名称**: `bert-base-uncased`  
- **多 GPU 支持**: 通过 `nn.DataParallel(model)` 实现，当检测到多块 GPU 时自动并行。

## 环境依赖

主要依赖库如下（请根据需求进行相应调整）：

- Python 3.7+
- PyTorch >= 1.10
- Transformers >= 4.0
- Datasets
- scikit-learn
- onnx
- onnxruntime
- tqdm (可选，用于显示进度条)
  
## 脚本流程
## 1. 设定设备 (GPU / CPU)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- 代码会自动检测是否有可用的 GPU，若无则使用 CPU。
  ```python
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
  ```
- 如有多个可用GPU则使用多GPU训练。
## 2. 加载数据

- 调用 `load_dataset("amazon_polarity")` 获取数据集。  
- 为了加快训练，只选取了部分数据进行实验。  
- 使用 `train_test_split` 将训练数据再拆分为训练集与验证集。

## 3. 定义数据集类 (ReviewDataset)

- 使用 `BertTokenizer `对文本进行分词、截断、填充。
- 将文本与标签打包成 `Dataset` 对象，供` DataLoader` 使用。
## 4. 创建数据加载器 (DataLoader)

- batch_size 为 16，可根据显存和速度情况进行调整。
- 通过 `Dataloader`将 `Dataset` 中的数据按批次读取。
## 5. 定义模型与优化器

1. **模型**  
   使用 `BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)` 初始化 BERT 模型。

2. **优化器**  
   采用 `AdamW` 优化器，学习率为 `2e-5`。
## 6. 训练与验证

- 训练若干个 `epoch`，每个 `epoch` 结束后在`validation set`上计算损失和 F1 分数并输出。  
- 每个 `epoch` 结束后会保存一次模型`check point`（`.pth` 文件）。  
## 7. 模型导出 (ONNX)

- 调用 `torch.onnx.export `将训练好的 BERT 模型导出为 `bert_model.onnx`。
## 8. ONNX Runtime 推理

- 使用 `onnxruntime.InferenceSession `加载导出的 `ONNX` 模型。
- 通过 `onnx_infer `函数输入待预测文本，并获得预测结果。
## 9. 加载最佳check point并在test set评估

- 可读取已经保存的检查点文件（如 `bert_model_epoch_5.pth`）。
- 对`test set`进行推理，计算并输出 `F1-score` 分数等指标，用于评估模型最终性能。
## 如何调整参数

1. **数据相关**  
- `train_texts[:20000] `及 `test_texts[:5000]` 等可修改为更大规模或使用完整数据集。
- 在 `ReviewDataset` 中，`max_len` 为 128。可根据实际平均文本长度进行调整。
2. **训练轮数 (epochs)**
- `epochs = 5` 为示例值，可根据数据量和需求增减。
3. **Batch Size**
- 在 `DataLoader` 中的 `batch_size=16` 可根据 GPU 显存和速度需求进行调节。
4. **学习率 (learning rate)**
- 在 `optimizer = optim.AdamW(model.parameters(), lr=2e-5)` 中调整 `lr`已获得更好的结果。
5. **多GPU训练**
使用 `nn.DataParallel` 对多 GPU 并行训练进行支持。

