# %%
# Import library
import random
import os
import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import BertTokenizer, BertForSequenceClassification

from bert.util import *
from bert.data import *
from bert.model import *
from bert.metric import *

# %%
# train function
def train(config):
    #Get arguments
    mode = config.mode
    model = config.model
    
    data_fn = config.data_fn
    data_dir = config.data_dir
    ckpt_dir = config.ckpt_dir
    log_dir = config.log_dir
    result_dir = config.result_dir
    
    lr = config.lr
    batch_size = config.batch_size
    num_epoch = config.num_epoch
    
    max_length = config.max_length
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Print arguments
    print("mode: %s" % mode)
    print("model: %s" % model)

    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)

    print("data file name: %s" % data_fn)
    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)
    
    print("max_length: %d" % max_length)

    print("device: %s" % device)
    
    #Make directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    #Get dataset and loader
    train_loader, valid_loader = DataPreprocessing(config)
    dataiter = iter(train_loader)
    text, label = dataiter.next()

    print('\nText example : \n', text)
    print('\nLabel example : \n', label)

    #Get tokenizer
    tokenizer = BertTokenizer.from_pretrained(model)
    '''
    Huggingface에서 beomi/kcbert-base 모델을 사용했습니다.
    '''
    
    #Get model
    model = BertForSequenceClassification.from_pretrained(model, num_labels=2)
    model.to(device)
    '''
    Huggingface에서 beomi/kcbert-base 모델을 사용했습니다.
    '''
    
    #Set optimizer and criterion
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    #Set initial variables
    total_correct = 0.0
    total_len = 0.0
    running_loss = 0.0
    valid_running_loss = 0.0
    valid_correct = 0.0
    running_step = 0
    train_loss_list = []
    valid_loss_list = []
    epoch_list = []
    valid_accuracy_list = []
    best_valid_loss = float("inf")

    
    #Train loop
    model.train()
    for epoch in range(1, num_epoch+1):
        for text, label in train_loader:
            opt.zero_grad()        
            encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length=300) for t in text]
            padded_list =  [e + [0] * (300-len(e)) for e in encoded_list]
            sample = torch.tensor(padded_list)
            
            sample, labels = sample.to(device), label.to(device)
            outputs = model(sample, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            pred = torch.argmax(F.softmax(logits), dim=1)
            correct = pred.eq(labels)
            total_correct += correct.sum().item()
            total_len += len(labels)
            
            running_loss += loss.item()
            loss.backward()
            opt.step()        
            running_step += 1
            
            if running_step % 10 == 0:
                accuracy_temp = total_correct / total_len
                loss_temp = running_loss / total_len
                
                print('Epoch : %d / %d, Step : %d / %d, accuracy : %.2f, loss : %.4f' %(epoch, num_epoch, running_step, len(train_loader), accuracy_temp, loss_temp))
        
        #validation phase per epoch
        model.eval()
        with torch.no_grad():
            for text, label in valid_loader:
                encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length=300) for t in text]
                padded_list =  [e + [0] * (300-len(e)) for e in encoded_list] 
                sample = torch.tensor(padded_list)
                
                sample, labels = sample.to(device), label.to(device)
                outputs = model(sample, labels=labels)
                loss, logits = outputs.loss, outputs.logits
                
                valid_running_loss += loss.item()
                
                pred = torch.argmax(F.softmax(logits), dim=1)
                valid_correct += pred.eq(labels).sum().item()
                
            train_loss = running_loss / len(train_loader)
            valid_loss = valid_running_loss / len(valid_loader)
            valid_accuracy = valid_correct / (batch_size * len(valid_loader))
            
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            epoch_list.append(epoch)
            valid_accuracy_list.append(valid_accuracy)
            
            print('VALIDATION : ')
            print('Epoch : %d / %d, Valid-accuracy : %.2f, Valid-loss : %.4f' %(epoch, num_epoch, valid_accuracy, valid_loss))
            
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                save_checkpoint(os.path.join(ckpt_dir, 'model.pt'), model, best_valid_loss)
                save_metrics(os.path.join(result_dir, 'best_metrics.pt'), valid_accuracy, train_loss_list, valid_loss_list, epoch_list)
        
        total_correct = 0.0
        total_len = 0.0
        running_loss = 0.0
        valid_running_loss = 0.0
        valid_correct = 0.0
        running_step = 0
        
        model.train()
  
    save_metrics(os.path.join(result_dir, 'metrics.pt'),valid_accuracy_list, train_loss_list, valid_loss_list, epoch_list)
    print('훈련 종료!')
    
#%%
# evaluation function
def validation(config):
    #Get arguments
    mode = config.mode
    model = config.model
    
    data_fn = config.data_fn
    data_dir = config.data_dir
    ckpt_dir = config.ckpt_dir
    log_dir = config.log_dir
    result_dir = config.result_dir
    
    lr = config.lr
    batch_size = config.batch_size
    num_epoch = config.num_epoch
    
    max_length = config.max_length
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('\n ||validation mode|| \n')
    
    #Plot training and validation logs
    valid_accuracy_list, train_loss_list, valid_loss_list, epoch_list = load_metrics(os.path.join(result_dir, 'metrics.pt'))
    plt.figure(figsize=(15,5))
    plt.plot(epoch_list, train_loss_list, label='Train')
    plt.plot(epoch_list, valid_loss_list, label='Valid')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
    
    plt.figure(figsize=(15,5))
    plt.plot(epoch_list, valid_accuracy_list, label='Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'accuracy_curve.png'))

    #Get dataset and loader
    test_set = pd.read_csv(os.path.join(data_dir, 'split', 'test.csv'), encoding='utf-8')
    test_set = test_set.fillna('NaN')
    test_set = Datasets(test_set)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    #Get tokenizer
    tokenizer = BertTokenizer.from_pretrained(model)
    
    #Get model
    model = BertForSequenceClassification.from_pretrained(model, num_labels=2)
    model = model.to(device)
    
    #Load best model
    load_checkpoint(os.path.join(ckpt_dir, 'model.pt'), model)
    
    #evaluation loop
    logit = []
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for text, label in test_loader:
            encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length=300) for t in text]
            padded_list =  [e + [0] * (300-len(e)) for e in encoded_list]
        
            sample = torch.tensor(padded_list)
            labels = label
            sample, labels = sample.to(device), labels.to(device)
            output = model(sample, labels=labels)
            output = output.logits
            
            logit.extend(output[:, -1].tolist())
            y_pred.extend(torch.argmax(F.softmax(output), dim=1).tolist())
            y_true.extend(labels.tolist())
        
        AUROC_score = roc_auc_score(y_true, y_pred)
        fprs, tprs, thresholds = roc_curve(y_true, logit)
        
    plt.figure(figsize=(15,5))
    plt.plot([0,1],[0,1])
    plt.plot(fprs, tprs, label='AUROC = %.6f' %AUROC_score)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(result_dir, 'ROC.png'))
    
def test(config):
    #Get arguments
    mode = config.mode
    model = config.model
    
    data_fn = config.data_fn
    data_dir = config.data_dir
    ckpt_dir = config.ckpt_dir
    log_dir = config.log_dir
    result_dir = config.result_dir
    
    lr = config.lr
    batch_size = config.batch_size
    num_epoch = config.num_epoch
    
    max_length = config.max_length
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('\n ||test mode|| \n')

    #Get dataset and loader
    test_loader = DataPreprocessing(config)
    
    #Get tokenizer
    tokenizer = BertTokenizer.from_pretrained(model)
    
    #Get model
    model = BertForSequenceClassification.from_pretrained(model, num_labels=2)
    model = model.to(device)
    
    #Load best model
    load_checkpoint(os.path.join(ckpt_dir, 'model.pt'), model)
    
    #evaluation loop
    logit = []
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for text, label in test_loader:
            encoded_list = [tokenizer.encode(t, add_special_tokens=True, max_length=300) for t in text]
            padded_list =  [e + [0] * (300-len(e)) for e in encoded_list]
        
            sample = torch.tensor(padded_list)
            labels = label
            sample, labels = sample.to(device), labels.to(device)
            output = model(sample, labels=labels)
            output = output.logits
            
            print(output)
            
            logit.extend(output[:, -1].tolist())
            y_pred.extend(torch.argmax(F.softmax(output), dim=1).tolist())
            y_true.extend(labels.tolist())
        
        AUROC_score = roc_auc_score(y_true, y_pred)
        fprs, tprs, thresholds = roc_curve(y_true, logit)
        
    plt.figure(figsize=(15,5))
    plt.plot([0,1],[0,1])
    plt.plot(fprs, tprs, label='AUROC = %.6f' %AUROC_score)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(result_dir, 'ROC.png'))