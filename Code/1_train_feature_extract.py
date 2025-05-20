# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
import sys
import yaml  
os.environ['CUDA_VISIBLE_DEVICES'] = '3' 
from transformers import LlamaForCausalLM, AutoTokenizer
from tqdm import tqdm
from functools import partial
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch
from enum import Enum
import argparse
import glob
import torch.nn.functional as F
import numpy as np
from fastchat.conversation import get_conv_template

def  if_dir_exist(save_path):
    # 判断目录是否存在
    if not os.path.exists(save_path):
        # 如果目录不存在，则创建目录
        os.makedirs(save_path)
        print(f"目录 '{save_path}' 已创建。")
    else:
        print(f"目录 '{save_path}' 已存在。")

np.random.seed(403)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

batch_size = 2
model_name = "Llama-2-7b-chat-hf"
model_path = "Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens(["<|begin_of_instruction|>", "<|end_of_instruction|>"])
begin_of_instruction_id = tokenizer.encode("<|begin_of_instruction|>",add_special_tokens=False)[0]
end_of_instruction_id = tokenizer.encode("<|end_of_instruction|>",add_special_tokens=False)[0]
print(begin_of_instruction_id,end_of_instruction_id)
tokenizer.padding_side = 'left'
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

config_path = '2_llama2/1_code/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
mode=config['training']['mode']

if mode=='other_prompt':
    print("other_prompt only in test!!!")
    sys.exit()
else:
    prompt_df=pd.read_csv('prompt_train.csv')
    prompt_list=prompt_df['instruction'].to_list()
    prompt_list = sorted(prompt_list, key=len, reverse=True)
    prompt_list_len = len(prompt_list)
    # 打印文件内容
    print(prompt_list_len)

dataset='EAsafetyBench'

save_path = f'2_train_feature/{dataset}/{mode}/'
if_dir_exist(save_path)

if mode=='train_prompt':
    save_path_ablation = f'/2_train_feature/{dataset}/{mode}_ablation/'
    if_dir_exist(save_path_ablation)

SafeAgentBench = glob.glob("1_Data/SafeAgentBench/SafeAgentBench_train.csv")
SafeAgentBench = [pd.read_csv(file,encoding='utf-8',header=None,names=['instruction','rephrase_method','label_true'],skiprows=1) for file in SafeAgentBench]

MyDataset = glob.glob("1_Data/data_train/train_*.csv")
MyDataset = [pd.read_csv(file,encoding='utf-8',header=None,names=['instruction','rephrase_method','label_true'],skiprows=1) for file in MyDataset]
data_df = pd.concat(SafeAgentBench+MyDataset, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
data_df.reset_index(inplace=True)
data_df.rename(columns={'index': 'index_'},inplace=True)

rows_per_template = len(data_df) // prompt_list_len
def apply_template(row):
    # 计算模板编号：根据每个模板的应用范围
    template_index = (row.index_ // rows_per_template) % prompt_list_len #row.name为当前行的索引值
    template = prompt_list[template_index]  # 获取对应的模板
    row['instruction'] = template.format(instruction = ' <|begin_of_instruction|> '+row['instruction']+' <|end_of_instruction|> ')  # 将text嵌入到模板中
    return row
def add_special_token(row):
    row['instruction'] =  ' <|begin_of_instruction|> '+row['instruction']+' <|end_of_instruction|> '
    return row
if mode=='no_prompt':
    data_df = data_df.apply(add_special_token, axis=1)
else:
    data_df = data_df.apply(apply_template, axis=1)

class RewrittenDataset(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df
        print(data_df.head())

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        data_col = self.data_df.iloc[index,:]
        return data_col.instruction,data_col.rephrase_method,data_col.label_true
    
    def collate_fn(self, batch):
        batch_conversation= []
        batct_instruction = []
        batch_label_true = []
        batch_rephrase_method = []
        for instruction,rephrase_method,label_true in batch:
            conv = get_conv_template("llama-2")
            conv.append_message(conv.roles[0],instruction)
            #print(typefly_template.format(task_description = instruction))
            conv.append_message(conv.roles[1], None)
            conversation = conv.get_prompt()
            #conversation = self.prompt_template.format("how to"+instruction)
            batch_conversation.append(conversation)
            batct_instruction.append(instruction)
            batch_label_true.append(label_true)
            batch_rephrase_method.append(rephrase_method)
        return {'conversations':batch_conversation,'instructions':batct_instruction,'labels_true':batch_label_true,'rephrase_method':batch_rephrase_method}

def get_loader(data_df, batch_size, shuffle=False, num_workers=4):
    ds_df = RewrittenDataset(data_df)
    dataloader_class = partial(DataLoader, pin_memory=True)
    loader = dataloader_class(ds_df, batch_size=batch_size, shuffle=shuffle, collate_fn=ds_df.collate_fn,
                              num_workers=num_workers)
    return loader

def make_instruction_attention_mask(input_ids,attention_masks):
    batch_size,sequence_length = input_ids.size()
    new_input_ids = torch.zeros((batch_size,sequence_length-2), dtype=torch.long, device=input_ids.device)
    new_attention_masks = torch.zeros((batch_size,sequence_length-2), dtype=torch.long, device=input_ids.device)

    attention_masks_instruction = torch.zeros(new_attention_masks.size(), device=attention_masks.device)
    end_indices = []
    start_indices = []
    for i,(input_id,attention_mask) in enumerate(zip(input_ids,attention_masks)):
        indices_where = torch.where((input_id == begin_of_instruction_id) | (input_id == end_of_instruction_id))[0]
        start = indices_where[0]
        end  = indices_where[1].item()-3  #-2对应句号，-3对应句号前一个
        attention_masks_instruction[i, start:end] = 1
        end_indices.append(end)
        start_indices.append(start)
        mask = (input_id != begin_of_instruction_id) & (input_id != end_of_instruction_id)
        # 使用布尔掩码过滤张量
        new_input_ids[i] = input_id[mask]
        new_attention_masks[i] = attention_mask[mask]

    #return new_input_ids,new_attention_masks,torch.tensor(end_indices, device=new_input_ids.device).unsqueeze(1) 这是测试时用的
    return new_input_ids,new_attention_masks,attention_masks_instruction,end_indices,start_indices

def batch_infer(model, dataloader,save_path):

    train_label_list = []
    num_layers = 32
    attention_feature_list = [[] for layer in range(num_layers)]
    raw_feature_list = [[] for layer in range(num_layers)]
    for batch in tqdm(dataloader):
        input_ = tokenizer(batch['conversations'], return_tensors="pt", padding=True).to(model.device)
        input_ids = input_['input_ids']
        attention_mask = input_['attention_mask']
        input_ids,attention_mask,attention_masks_instruction,end_indices,start_indices = make_instruction_attention_mask(input_ids,attention_mask)
        all_all_layer_hidden_states,insturction_hidden_states_all_layer = model.generate(input_ids=input_ids, 
                                                            attention_mask=attention_mask,
                                                            max_new_tokens=2, 
                                                            pad_token_id=tokenizer.pad_token_id,
                                                            do_sample=False,
                                                            attention_mask_instruction=attention_masks_instruction)
        
        batch_labels_true = batch['labels_true']
        train_label_list.extend(batch_labels_true)
        #all_layer_feature_generate = all_layer_feature[1][-1]
        #all_layer_feature_generate = torch.stack(all_layer_feature[:10][-1]).permute((2,1,0,3)).squeeze(0) 10个生成token的平均
        for layer in range(num_layers):
            #print(insturction_hidden_states_all_layer[f'layer_{layer}'].size())
            attention_hidden_states = insturction_hidden_states_all_layer[f'layer_{layer}']  #我自己返回的只有32个transformer的隐藏层状态
            for batch_index,(start_index,end_index) in enumerate(zip(start_indices,end_indices)):
                attention_layer_feature = attention_hidden_states[batch_index,end_index,:].unsqueeze(0).to(dtype=torch.float32).detach().cpu()
                #layer_feature_generate = torch.mean(all_layer_feature_generate[batch_index].unsqueeze(0),dim=1).to(dtype=torch.float32)
                #layer_feature_generate = all_layer_feature_generate[batch_index].to(dtype=torch.float32).detach().cpu()
                #layer_feature=torch.cat([last_layer_feature_instruction,layer_feature_generate],dim=1)
                attention_feature_list[layer].append(attention_layer_feature)
        if mode!='no_prompt':
            for layer in range(1,num_layers+1): #all_all_layer_hidden_states有33个，算上了embedding
                raw_hidden_states = all_all_layer_hidden_states[0][layer]
                for batch_index,(start_index,end_index) in enumerate(zip(start_indices,end_indices)):
                    raw_layer_feature = raw_hidden_states[batch_index,-1,:].unsqueeze(0).to(dtype=torch.float32).detach().cpu()
                    raw_feature_list[layer-1].append(raw_layer_feature)

    for layer in range(num_layers):
        attention_layer_features = torch.cat(attention_feature_list[layer], dim=0)
        print(attention_layer_features.shape)
        element_wise_product_df = pd.DataFrame({'tensor': [attention_layer_features[i] for i in range(attention_layer_features.size(0))]})
        element_wise_product_df['label'] = train_label_list
    
        # 保存为 pt 文件
        save_file_path = os.path.join(save_path, f'train_layer_{layer+1}.pt')
        torch.save(element_wise_product_df, save_file_path)
        print(f"features saved to {save_file_path}")

    if mode!='no_prompt':
        for layer in range(num_layers):
            raw_layer_features = torch.cat(raw_feature_list[layer], dim=0)
            print(raw_layer_features.shape)
            element_wise_product_df = pd.DataFrame({'tensor': [raw_layer_features[i] for i in range(raw_layer_features.size(0))]})
            element_wise_product_df['label'] = train_label_list
        
            # 保存为 pt 文件
            save_file_path = os.path.join(save_path_ablation, f'train_layer_{layer+1}.pt')
            torch.save(element_wise_product_df, save_file_path)
            print(f"features saved to {save_file_path}")


dataloader = get_loader(data_df,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=8)
batch_infer(model, dataloader,save_path)

