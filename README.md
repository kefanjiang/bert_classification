# Amazon Review Sentiment Analysis (BERT)

## Overview
This project implements a BERT-based text classification model to analyze the sentiment of Amazon product reviews. The model classifies reviews as **positive** or **negative** using the Amazon Polarity dataset. The pipeline includes data preprocessing, fine-tuning BERT, model evaluation, and exporting to ONNX for optimized inference.

## Features
- **Fine-tuned BERT for sentiment classification** with PyTorch and Hugging Face Transformers
- **Amazon Polarity dataset preprocessing** and handling of imbalanced data
- **Multi-GPU training with `nn.DataParallel`** for efficiency
- **Evaluation with F1-score** achieving **0.9176** on the test set
- **ONNX model export** for optimized inference in production

## Installation
To set up the environment, install the required dependencies:
```bash
pip install torch transformers datasets onnx onnxruntime scikit-learn
```

## Usage
### 1. Data Preparation
Load the Amazon Polarity dataset and preprocess it:
```python
from datasets import load_dataset
dataset = load_dataset("amazon_polarity")
```

### 2. Model Training
Fine-tune BERT on the dataset:
```python
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
```

### 3. Model Evaluation
Evaluate performance using F1-score:
```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1-Score: {f1:.4f}")
```

### 4. ONNX Conversion
Convert the model for optimized inference:
```python
import torch
import onnx

dummy_input = torch.ones(1, 512, dtype=torch.long)
torch.onnx.export(model, dummy_input, "bert_model.onnx")
```

### 5. ONNX Inference
Load and run inference with ONNX:
```python
import onnxruntime as ort
session = ort.InferenceSession("bert_model.onnx")
```

## Results
- Achieved an **F1-score of 0.9176** on the test set
- ONNX model significantly improved inference speed






