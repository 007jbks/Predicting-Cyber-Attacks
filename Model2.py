import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import torch
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset

MODEL_DIR = "/kaggle/input/huggingface-bert/"

# Load and preprocess data
df = pd.read_csv('../input/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv', low_memory=False)

drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4", 
                "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", 
                "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport", 
                "tcp.dstport", "udp.port", "mqtt.msg"]

df.drop(drop_columns, axis=1, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df.drop_duplicates(subset=None, keep="first", inplace=True)
df = shuffle(df)

def one_hot_encode(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

# One hot encoding
one_hot_encode(df, 'http.request.method')
one_hot_encode(df, 'http.referer')
one_hot_encode(df, "http.request.version")
one_hot_encode(df, "dns.qry.name.len")
one_hot_encode(df, "mqtt.conack.flags")
one_hot_encode(df, "mqtt.protoname")
one_hot_encode(df, "mqtt.topic")

# Save preprocessed data
df.to_csv('preprocessed_DNN.csv', encoding='utf-8', index=False)

# Load preprocessed data
df = pd.read_csv('preprocessed_DNN.csv', low_memory=False)
feat_cols = list(df.columns)
label_col = "Attack_type"
feat_cols.remove(label_col)

X = df[feat_cols].astype(str)
y = df[label_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR + "bert-large-uncased")

def tokenize_data(text_data):
    return tokenizer(text_data, padding=True, truncation=True, max_length=512, return_tensors='pt')

# Tokenize text data
X_train_tokens = [tokenize_data(text) for text in X_train.apply(lambda row: ' '.join(row), axis=1)]
X_test_tokens = [tokenize_data(text) for text in X_test.apply(lambda row: ' '.join(row), axis=1)]

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Prepare datasets
train_dataset = TensorDataset(torch.cat([item['input_ids'] for item in X_train_tokens], dim=0), torch.tensor(y_train))
test_dataset = TensorDataset(torch.cat([item['input_ids'] for item in X_test_tokens], dim=0), torch.tensor(y_test))

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Load the model for sequence classification
model = BertForSequenceClassification.from_pretrained(MODEL_DIR + "bert-large-uncased", num_labels=len(label_encoder.classes_))
optimizer = AdamW(model.parameters(), lr=1e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training
epochs = 4
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        b_input_ids, b_labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(b_input_ids, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1} complete')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids, b_labels = [x.to(device) for x in batch]
        outputs = model(b_input_ids)
        _, predicted = torch.max(outputs.logits, 1)
        total += b_labels.size(0)
        correct += (predicted == b_labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')
