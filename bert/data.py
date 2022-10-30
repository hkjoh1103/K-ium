# %%
# Import library
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader

# %%
class Datasets(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])
        return text, label

def DataPreprocessing(config):
    data_fn = config.data_fn
    data_dir = config.data_dir
    batch_size = config.batch_size
    
    split_dir = os.path.join(data_dir, 'split')
    
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        
        df = pd.read_csv(data_fn, encoding='utf-8')
        df = df.iloc[:, 1:3]
        
        train_set, test_set = train_test_split(df, test_size=0.1, shuffle=True, random_state=221030)
        
        train_set.to_csv(os.path.join(split_dir, 'train.csv'), encoding='utf-8', index=False)
        test_set.to_csv(os.path.join(split_dir, 'test.csv'), encoding='utf-8', index=False)
        
    else:
        train_set = pd.read_csv(os.path.join(split_dir, 'train.csv'), encoding='utf-8')
    
    train_set, valid_set = train_test_split(train_set, test_size=0.1, shuffle=True, random_state=221030)
    
    train_set = Datasets(train_set)
    valid_set = Datasets(valid_set)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    return train_loader, valid_loader
