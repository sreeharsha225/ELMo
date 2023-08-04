import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#dataloaders
from torch.utils.data import DataLoader
import numpy as np
import pickle
import json
import sys
import re
import tqdm
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


EmbeddingSize=100
HiddenSize=256
Batch_size=100
Epochs=10
Learning_rate=0.001

# elmo model
class ElmoModel(nn.Module):
    def __init__(self,embedding_matrix, vocab_size,embedding_dim=EmbeddingSize, hidden_dim = HiddenSize, num_layers=2, dropout_prob=0.2):
        super(ElmoModel,self).__init__()
        # Word embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(embedding_matrix))  # initialize word embeddings
        
        # Bi-LSTM layers
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            dropout=dropout_prob,
                            bidirectional=True,
                            batch_first=True)
        
        # Linear projection layers
        self.linear_1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.linear_2 = nn.Linear(hidden_dim*2, 1, bias=False)
    
    def forward(self, input):
        # Word embedding look up
        embeds = self.word_embeddings(input)
        
        # Bi-LSTM
        lstm_out, _ = self.lstm(embeds)
        
        # Linear projection
        linear_out = self.linear_1(lstm_out)
        linear_out = self.linear_2(linear_out)
        
        # Sigmoid
        prob = torch.sigmoid(linear_out)
        return prob

def tokenize(text):
    # tokenize
    tokens=word_tokenize(text.lower())
    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(w) for w in tokens]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    return tokens

def preProcessMNLI(data):
    total_data={'premise':[],'hypothesis':[],'label':[]}
    train_data={'premise':[],'hypothesis':[],'label':[]}
    for d in data['train']:
        train_data['premise'].append(tokenize(d['premise']))
        train_data['hypothesis'].append(tokenize(d['hypothesis']))
        train_data['label'].append(d['label'])
        total_data['premise'].append(tokenize(d['premise']))
        total_data['hypothesis'].append(tokenize(d['hypothesis']))
        total_data['label'].append(d['label'])

    
    dev_data={'premise':[],'hypothesis':[],'label':[]}
    for d in data['validation_matched']:
        dev_data['premise'].append(tokenize(d['premise']))
        dev_data['hypothesis'].append(tokenize(d['hypothesis']))
        dev_data['label'].append(d['label'])
        total_data['premise'].append(tokenize(d['premise']))
        total_data['hypothesis'].append(tokenize(d['hypothesis']))
        total_data['label'].append(d['label'])
    
    test_data={'premise':[],'hypothesis':[],'label':[]}
    for d in data['validation_mismatched']:
        test_data['premise'].append(tokenize(d['premise']))
        test_data['hypothesis'].append(tokenize(d['hypothesis']))
        test_data['label'].append(d['label'])
        total_data['premise'].append(tokenize(d['premise']))
        total_data['hypothesis'].append(tokenize(d['hypothesis']))
        total_data['label'].append(d['label'])

    return train_data,dev_data,test_data,total_data

def preProcessSST(data):
    total_data={'sentence':[],'label':[]}
    train_data={'sentence':[],'label':[]}
    for d in data['train']:
        train_data['sentence'].append(tokenize(d['sentence']))
        train_data['label'].append(d['label'])
        total_data['sentence'].append(tokenize(d['sentence']))
        total_data['label'].append(d['label'])


    dev_data={'sentence':[],'label':[]}
    for d in data['validation']:
        dev_data['sentence'].append(tokenize(d['sentence']))
        dev_data['label'].append(d['label'])
        total_data['sentence'].append(tokenize(d['sentence']))
        total_data['label'].append(d['label'])
    
    test_data={'sentence':[],'label':[]}
    for d in data['test']:
        test_data['sentence'].append(tokenize(d['sentence']))
        test_data['label'].append(d['label'])
        total_data['sentence'].append(tokenize(d['sentence']))
        total_data['label'].append(d['label'])
    
    return train_data,dev_data,test_data,total_data

#embed matrix from glove by calculating word2idx
def getEmbedMatrix():
    #load glove file
    word2idx={"":0,"<UNK>":1}
    with open('glove.6B.100d.txt', 'r') as f:
        glove = {}
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            glove[word] = vec
    #create embedding matrix
    embedding_matrix = np.zeros((len(glove)+2, EmbeddingSize))
    for i, word in enumerate(glove) :
        embedding_matrix[i+2] = glove[word]
        word2idx[word] = i+2
    return embedding_matrix,word2idx

#convert dataset to index
def convertsstDatasetToIndex(dataset,word2idx):
    for i in range(len(dataset['sentence'])):
        for j in range(len(dataset['sentence'][i])):
            if dataset['sentence'][i][j] in word2idx:
                dataset['sentence'][i][j]=word2idx[dataset['sentence'][i][j]]
            else:
                dataset['sentence'][i][j]=word2idx['<UNK>']
    return dataset

#convert dataset to tensor
def convertDatasetToTensor(dataset):
    for i in range(len(dataset['sentence'])):
        dataset['sentence'][i]=torch.tensor(dataset['sentence'][i])
    return dataset


#convert dataset to index
def convertmnliDatasetToIndex(dataset,word2idx):
    for i in range(len(dataset['premise'])):
        for j in range(len(dataset['premise'][i])):
            if dataset['premise'][i][j] in word2idx:
                dataset['premise'][i][j]=word2idx[dataset['premise'][i][j]]
            else:
                dataset['premise'][i][j]=word2idx['<UNK>']
        for j in range(len(dataset['hypothesis'][i])):
            if dataset['hypothesis'][i][j] in word2idx:
                dataset['hypothesis'][i][j]=word2idx[dataset['hypothesis'][i][j]]
            else:
                dataset['hypothesis'][i][j]=word2idx['<UNK>']
    return dataset

# train the model
def train(model,train_data,dev_data,optimizer,criterion):
    model.train()
    total_loss=0
    for i in range(len(train_data['sentence'])):
        optimizer.zero_grad()
        output=model(train_data['sentence'][i],train_data['label'][i])
        loss=criterion(output,train_data['label'][i])
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
    return total_loss/len(train_data['sentence'])



#main function
def main():
    #loading datasets
    #load sst dataset
    sst_dataset = load_dataset('sst')
    #load multi_nli dataset
    multi_nli_dataset = load_dataset('multi_nli')

    #preprocess datasets
    #preprocess sst dataset
    sst_train_data,sst_dev_data,sst_test_data,sst_total_data=preProcessSST(sst_dataset)
    #preprocess multi_nli dataset
    multi_nli_train_data,multi_nli_dev_data,multi_nli_test_data,multi_nli_total_data=preProcessMNLI(multi_nli_dataset)

    #get embedding matrix
    embedding_matrix,word2idx=getEmbedMatrix()

    #convert dataset to index
    sst_train_data=convertsstDatasetToIndex(sst_train_data,word2idx)
    sst_dev_data=convertsstDatasetToIndex(sst_dev_data,word2idx)
    sst_test_data=convertsstDatasetToIndex(sst_test_data,word2idx)
    multi_nli_train_data=convertmnliDatasetToIndex(multi_nli_train_data,word2idx)
    multi_nli_dev_data=convertmnliDatasetToIndex(multi_nli_dev_data,word2idx)
    multi_nli_test_data=convertmnliDatasetToIndex(multi_nli_test_data,word2idx)

    #convert dataset to tensor
    sst_train_data=convertDatasetToTensor(sst_train_data)
    sst_dev_data=convertDatasetToTensor(sst_dev_data)
    sst_test_data=convertDatasetToTensor(sst_test_data)
    multi_nli_train_data=convertDatasetToTensor(multi_nli_train_data)
    multi_nli_dev_data=convertDatasetToTensor(multi_nli_dev_data)
    multi_nli_test_data=convertDatasetToTensor(multi_nli_test_data)

    #create dataloader
    sst_train_dataloader = DataLoader(sst_train_data, batch_size=Batch_size, shuffle=True)
    sst_dev_dataloader = DataLoader(sst_dev_data, batch_size=Batch_size, shuffle=True)
    sst_test_dataloader = DataLoader(sst_test_data, batch_size=Batch_size, shuffle=True)
    multi_nli_train_dataloader = DataLoader(multi_nli_train_data, batch_size=Batch_size, shuffle=True)
    multi_nli_dev_dataloader = DataLoader(multi_nli_dev_data, batch_size=Batch_size, shuffle=True)
    multi_nli_test_dataloader = DataLoader(multi_nli_test_data, batch_size=Batch_size, shuffle=True)

    #Initialize model
    model = ElmoModel(embedding_matrix,len(word2idx), EmbeddingSize, HiddenSize)

    #Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)

    #Initialize loss function
    criterion = nn.CrossEntropyLoss()


    
    


    
    

    


if __name__ == '__main__':
    main()
