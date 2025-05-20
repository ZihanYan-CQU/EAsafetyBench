import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam,AdamW
from torch.utils.data import DataLoader,Dataset
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method


DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def is_directory_has_six_files(directory_path):
    return len(os.listdir(directory_path)) == 6

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ThreeLayerClassifier(nn.Module):  
    def __init__(self, dim):
        super(ThreeLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(dim , 1024)
        self.fc2 = nn.Linear(1024 , 512)
        self.fc3 = nn.Linear(512 , 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x           
    
class TrainDataset(Dataset):
    def __init__(self,feature_path_list):
        self.feature = pd.concat([torch.load(feature_path,weights_only=False) for feature_path in feature_path_list],axis=0)
        #self.feature = self.feature.sample(n=20)

        #stratified_sample = self.feature.groupby('label').apply(lambda x: x.sample(n=100))
        #self.feature = stratified_sample.reset_index(drop=True)
        #self.features = self.features[self.features['label']!=4]
        print(len(self.feature))
        print(self.feature.head())

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):
        data_col = self.feature.iloc[index,:]
        return data_col.tensor,torch.tensor(data_col.label, dtype=torch.long)

    def collate_fn(self, batch):
        batch_tensor= []
        batct_label = []
        for tensor,label in batch:
            batch_tensor.append(tensor.cuda())
            if label == 0:
                batct_label.append(0)
            else:
                batct_label.append(1)
        batch_tensor = torch.stack(batch_tensor)
        batct_label = torch.tensor(batct_label)
        return {'features':batch_tensor,'labels':batct_label}

def get_loader(feature_path_list, batch_size, shuffle=False, num_workers=4):
    ds_df = TrainDataset(feature_path_list)
    #print(ds_df.labels)
    dataloader_class = partial(DataLoader, pin_memory=False)
    loader = dataloader_class(ds_df, batch_size=batch_size, shuffle=shuffle, collate_fn=ds_df.collate_fn,
                              num_workers=num_workers)
    return loader
    
def batch_train(model, dataloader,lr,weight_decay,epochs,model_save_path,dir_path):

    optimizer = Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    StepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,100], gamma=0.1)
    best_loss=1
    model.train()
    criterion = nn.CrossEntropyLoss()
    max_epoch = max(epochs)
    for epoch in range(1,max_epoch+1):  #epochs=4，range(1,epochs+1)=(1,5),1个epoch等于使用训练集中的全部样本训练一次
        tloss = []
        bar = tqdm(dataloader)
        numbers = 0
        for index,batch in enumerate(bar):
            features = batch['features']
            labels = batch['labels']
            numbers += len(labels)
            optimizer.zero_grad()
            y_pred = torch.squeeze(model(features),1)
            loss = criterion(y_pred.to(DEVICE),labels.to(DEVICE))
            sloss=loss
            sloss.backward()  #反向传播，计算当前梯度
            tloss.append(sloss.item())   #梯度储存到tloss中
            optimizer.step()
            bar.set_postfix(epoch=epoch,learning_rate=optimizer.param_groups[0]['lr'],loss=np.array(tloss).mean())
        StepLR.step()
        print(numbers)
        if np.array(tloss).mean()<best_loss:
            torch.save(model.state_dict(),dir_path+'lowest_loss.pt')
            best_loss = np.array(tloss).mean()
        if epoch in epochs:
            torch.save(model.state_dict(),model_save_path.format(epoch))

def train_for_layer_index(weight_decay,epochs,lr,batch_size,mode,layer_indices,dataset):
    shuffle = True
    for layer_index in layer_indices:
        feature_path_list = [f'2_llama2/2_train_feature/{dataset}/{mode}/train_layer_{layer_index}.pt']
        model_save_path = f"2_llama2/4_model/{dataset}/{mode}/layer_{layer_index}/b{batch_size}_lr{'{:.0e}'.format(lr)}_e{{}}.pt"
        dir_path = f"2_llama2/4_model/{dataset}/{mode}/layer_{layer_index}/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"目录 '{dir_path}' 已创建。")
        else:
            print(f"目录 '{dir_path}' 已存在。")

        if is_directory_has_six_files(dir_path):
            print(f'{mode} layer {layer_index} 已完成训练')
            continue

        model = ThreeLayerClassifier(dim=4096)
        model.to(DEVICE)

        dataloader = get_loader(feature_path_list=feature_path_list,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=0)

        batch_train(model, dataloader, lr, weight_decay, epochs, model_save_path,dir_path)

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    seed = 403
    set_seed(seed)

    config_path = '2_llama2/1_code/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    batch_size = config['train_MLP']['batch_size']
    lr = config['train_MLP']['learning_rate']
    weight_decay = config['train_MLP']['weight_decay']
    epochs = config['train_MLP']['epochs']
    layer_indicies = config['train_MLP']['layer_index']
    mode_list = config['train_MLP']['mode']

    dataset='EAsafetyBench'

    for mode in mode_list:
        print('------------------------------------------------------------------mode: ', mode)
        if layer_indicies == 'all':
            folder_path = f'2_llama2/2_train_feature/{dataset}/train_prompt'
            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            layer_indicies = [i + 1 for i in range(file_count)]

        # Split the layer indices for parallel training
        half = len(layer_indicies) // 2
        layer_indices_part1 = layer_indicies[:half]
        layer_indices_part2 = layer_indicies[half:]

        # Use multiprocessing to train in parallel
        with Pool(processes=2) as pool:
            pool.starmap(train_for_layer_index, [(weight_decay,epochs,lr,batch_size,mode,layer_indices_part1,dataset), (weight_decay,epochs,lr,batch_size,mode,layer_indices_part2,dataset)])