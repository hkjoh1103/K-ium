# %%
# Import library
import os
import numpy as np

import torch
import torch.nn as nn

import torchvision.utils as utils

def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
      print("Invalid save path")
      return    
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):    
    if load_path==None:
      print("Invalid load path")
      return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def save_metrics(save_path, accuracy, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
      print("Invalid save path")
      return    
    state_dict = {'accuracy': accuracy,
                  'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):
    if load_path==None:
      print("Invalid load path")
      return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    return state_dict['accuracy'], state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']