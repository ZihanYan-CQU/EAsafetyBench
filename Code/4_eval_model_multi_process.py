import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
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
import glob
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def process_mask(data_df):
    data_df['y_pred'] = data_df['predict'].apply(lambda x: 0 if x==0 else 1)
    data_df['y_true'] = data_df['label_true'].apply(lambda x: 0 if x==0 else 1)
    y_pred = list(data_df['y_pred'])
    y_true = list(data_df['y_true'])
    f1 = f1_score(y_true, y_pred)
    return f1

def seed_everything(seed):   #设置各种种子，使得结果可复现
    random.seed(seed) #设置种子
    np.random.seed(seed)        #设置种子

def is_directory_has_six_files(directory_path):
    return len(os.listdir(directory_path)) == 6

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
    
class TestDataset(Dataset):
    def __init__(self,feature_path_list):
        self.feature = pd.concat([torch.load(feature_path,weights_only=False) for feature_path in feature_path_list],axis=0)

        #self.features = self.features[self.features['label']!=4]
        #print(len(self.feature))
        #print(self.feature.head())

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
    ds_df = TestDataset(feature_path_list)
    #print(ds_df.labels)
    dataloader_class = partial(DataLoader, pin_memory=False)
    loader = dataloader_class(ds_df, batch_size=batch_size, shuffle=shuffle, collate_fn=ds_df.collate_fn,
                              num_workers=num_workers)
    return loader,ds_df.__len__()
    
def batch_eval(moderator, dataloader,eval_template,save_path):
    predict_list = []
    safe_prob_list = []
    unsafe_prob_list = []
    safe_num = 0
    unsafe_num = 0
    for batch in tqdm(dataloader):
        pred_scores = torch.squeeze(moderator(batch['features'].to(torch.float32)),1)
        for batch_index, pred_score in enumerate(pred_scores):
            pred = pred_score.argmax(dim=0)
            if pred==1:
                pred_label = 1
                unsafe_num+=1
            else:
                pred_label = 0
                safe_num+=1
            pred_pro = F.softmax(pred_score, dim=0) 
            predict_list.append(pred_label)
            safe_prob_list.append(pred_pro[0].item())
            unsafe_prob_list.append(pred_pro[1].item())
    print("  safe: ",safe_num)
    print("unsafe: ",unsafe_num)
    eval_template['predict'] = predict_list
    eval_template['safe_prob'] = safe_prob_list
    eval_template['unsafe_prob'] = unsafe_prob_list
    f1 = process_mask(eval_template)
    print(f'---------------------------------------{save_path} F1-score:',f1)
    eval_template.to_csv(save_path)


def eval_for_layer_index(train_model,epochs,lr,train_batch_size,mode,layer_indicies,dataset):
    batch_size = 128
    time_list=[]
    for layer_index in layer_indicies:
        dir_path = f'2_llama2/5_eval/{dataset}/{mode}/layer_{layer_index}/'
        if not os.path.exists(dir_path):
            # 如果目录不存在，则创建目录
            os.makedirs(dir_path)
            print(f"目录 '{dir_path}' 已创建。")
        else:
            print(f"目录 '{dir_path}' 已存在。")

        if is_directory_has_six_files(dir_path)==True:
            print(f'layer {layer_index} 已完成评估')
            continue
        for epoch in epochs:
            moderator_name = f"b{train_batch_size}_lr{'{:.0e}'.format(lr)}_e{epoch}"

            eval_feature_path_list = [f'2_llama2/3_test_feature/{dataset}/{mode}/test_layer_{layer_index}.pt']

            save_path = f'2_llama2/5_eval/{dataset}/{mode}/layer_{layer_index}/{moderator_name}.csv'
            
            moderator_path = f'2_llama2/4_model/{dataset}/{train_model}/layer_{layer_index}/{moderator_name}.pt'

            moderator = ThreeLayerClassifier(dim=4096)
            moderator.load_state_dict(torch.load(moderator_path), assign = True)
            moderator.cuda()
            moderator.eval()

            eval_template =  pd.read_csv('2_llama2/1_code/test_sample.csv')
            dataloader,test_len = get_loader(feature_path_list=eval_feature_path_list,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=0)
            start_time = time.time()
            batch_eval(moderator, dataloader,eval_template,save_path)
            classify_test_time = time.time() - start_time
            #print(test_len)
            print('Average Classify Time:',classify_test_time / test_len)
            time_list.append(classify_test_time / test_len)
    pd.DataFrame({
        'Average Classify Time': time_list,
    }).to_csv('2_llama2/1_code/time.csv', index=False)

if __name__ == '__main__':
    set_start_method('spawn', force=True)
    seed = 403
    seed_everything(seed)

    config_path = '2_llama2/1_code/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    layer_indicies=config['train_MLP']['layer_index']
    epochs = config['train_MLP']['epochs']
    train_batch_size = config['train_MLP']['batch_size']
    lr = config['train_MLP']['learning_rate']
    mode_list=config['eval_model']['mode']

    dataset='EAsafetyBench'

    for mode in mode_list:
        print('------------------------------------------------------------------mode: ',mode)
        if layer_indicies=='all':
            folder_path=f'2_llama2/2_train_feature/{dataset}/train_prompt'
            file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            layer_indicies=[ i+1 for i in range(file_count)]

        if mode=='no_prompt':
            train_model='no_prompt'

        elif mode=='train_prompt': 
            train_model='train_prompt'
        elif mode=='other_prompt': 
            train_model='train_prompt'

        elif mode=='train_prompt_ablation': 
            train_model='train_prompt_ablation'
        elif mode=='other_prompt_ablation': 
            train_model='train_prompt_ablation'

        print('------------------------------------------------------------------eval_model_name: ',train_model)
        # Split the layer indices for parallel training
        half = len(layer_indicies) // 2
        layer_indices_part1 = layer_indicies[:half]
        print(layer_indices_part1)
        layer_indices_part2 = layer_indicies[half:]
        print(layer_indices_part2)
        # Use multiprocessing to train in parallel
        with Pool(processes=2) as pool:
            pool.starmap(eval_for_layer_index, [(train_model,epochs,lr,train_batch_size,mode,layer_indices_part1,dataset), (train_model,epochs,lr,train_batch_size,mode,layer_indices_part2,dataset)])
