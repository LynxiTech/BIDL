# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.
import os
import numpy as np
import torch
import random
from collections import Counter

datapath = r'./aclImdb'
save_dir = r'./data'

class ImdbDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocab, max_len=500, pad_idx=0):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = pad_idx
 
    def __len__(self):
        return len(self.texts)
 
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized_text = text[:self.max_len] + [self.pad_idx] * (self.max_len - len(text))
        return torch.tensor(tokenized_text, dtype=torch.long), torch.tensor(label, dtype=torch.float)
    
def get_data(datapath):
    pos_files = os.listdir(datapath + '/pos')
    neg_files = os.listdir(datapath + '/neg')
    print(f"Positive files: {len(pos_files)}")
    print(f"Negative files: {len(neg_files)}")

    pos_all = []
    neg_all = []
    for pf, nf in zip(pos_files, neg_files):
        with open(os.path.join(datapath, 'pos', pf), encoding='utf-8') as f:
            pos_all.append(f.read())
        with open(os.path.join(datapath, 'neg', nf), encoding='utf-8') as f:
            neg_all.append(f.read())

    X_orig = np.array(pos_all + neg_all)
    Y_orig = np.array([1] * len(pos_all) + [0] * len(neg_all))
    print("X_orig shape:", X_orig.shape)
    print("Y_orig shape:", Y_orig.shape)

    return X_orig, Y_orig
# 构建词汇表
def build_vocab(texts, vocab_size=999):
    word_counts = Counter()
    for text in texts:
        for word in text.split():
            word_counts[word] += 1
    vocab = {word: idx+1 for idx, (word, count) in enumerate(word_counts.most_common(vocab_size))}
    vocab['<unk>'] = vocab_size+1
    vocab['<pad>'] = 0
    return vocab
 
# 数据切分
def train_val_split(data, split_ratio=0.75):
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]
    return train_data, val_data
 
# 数据预处理
def preprocess_data(texts, labels, vocab, max_len=500):
    tokenized_texts = [[vocab.get(word, vocab['<unk>']) for word in text.split()] for text in texts]
    
    # Pad texts at the beginning
    padded_texts = [[vocab['<pad>']] * (max_len - len(text)) + text for text in tokenized_texts]
    
    # Ensure all sequences are truncated to max_len
    padded_texts = [text[:max_len] for text in padded_texts]
    return padded_texts, labels

def generate_train_data():
    X_train_orig, Y_train_orig = get_data(os.path.join(datapath, 'train'))
    X_test_orig, Y_test_orig = get_data(os.path.join(datapath, 'test'))

    texts = np.concatenate([X_train_orig, X_test_orig])
    labels = np.concatenate([Y_train_orig, Y_test_orig])

    vocab = build_vocab(texts)
    print(f"Vocabulary size: {len(vocab)}")
 
    # 切分训练集和验证集
    data = list(zip(texts, labels))
    train_data, val_data = train_val_split(data, split_ratio=0.75)
    X_train, y_train = zip(*train_data)
    X_val, y_val = zip(*val_data)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
 
    # 预处理训练集和验证集
    train_texts, train_labels = preprocess_data(X_train, y_train, vocab, max_len=500)
    val_texts, val_labels = preprocess_data(X_val, y_val, vocab, max_len=500)
 
    train_dataset = ImdbDataset(train_texts, train_labels, vocab)
    val_dataset = ImdbDataset(val_texts, val_labels, vocab)
 
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
 
    return train_dataset, val_dataset#, vocab