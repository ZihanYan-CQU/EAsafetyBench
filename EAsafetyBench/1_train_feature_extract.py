# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
import yaml
import glob
from tqdm import tqdm
from functools import partial
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, AutoTokenizer
from fastchat.conversation import get_conv_template
from pathlib import Path
# 1. 获取脚本所在目录的绝对路径
root_dir = Path(__file__).resolve().parent
# 2. 切换工作目录
os.chdir(root_dir)

# 验证
print(f"当前工作目录已设置为: {os.getcwd()}")

# ==============================================================================
# ============================== 配置区域 (CONFIG) =============================
# ==============================================================================

# --- 1. 环境与基础设置 ---
os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # 指定GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RANDOM_SEED = 403

# --- 2. 模型配置 ---
MODEL_PATH = "/mnt/data/YZH/weight/Llama-3.2-1B-Instruct"
NUM_HIDDEN_LAYERS = 16  # 模型层数

# 特殊 Token 定义
SPECIAL_TOKENS = {
    "begin": "<|begin_of_instruction|>",
    "end": "<|end_of_instruction|>"
}

# --- 3. 数据路径配置 ---
BASE_EXP_DIR = "./"
PROMPT_MODE = 'prompt_train'
PROMPT_TRAIN_CSV = os.path.join(BASE_EXP_DIR, f"train/{PROMPT_MODE}.csv")
DATA_TRAIN_GLOB = os.path.join(BASE_EXP_DIR, "train/train_*.csv")

# --- 4. 保存路径配置 ---
FEATURE_SAVE_DIR = os.path.join(BASE_EXP_DIR, f"train_feature/{PROMPT_MODE}")

# --- 5. 训练超参数 ---
BATCH_SIZE = 2

# ==============================================================================
# ============================ 逻辑代码 (LOGIC) ===============================
# ==============================================================================

def if_dir_exist(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"目录 '{save_path}' 已创建。")
    else:
        print(f"目录 '{save_path}' 已存在。")

# 设置随机种子
np.random.seed(RANDOM_SEED)

# 构建保存路径
if_dir_exist(FEATURE_SAVE_DIR)

# ---------------------- 加载 Tokenizer & Model ----------------------
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens([SPECIAL_TOKENS["begin"], SPECIAL_TOKENS["end"]])

# 获取特殊Token ID
begin_of_instruction_id = tokenizer.encode(SPECIAL_TOKENS["begin"], add_special_tokens=False)[0]
end_of_instruction_id = tokenizer.encode(SPECIAL_TOKENS["end"], add_special_tokens=False)[0]
print(f"Special Token IDs -> Begin: {begin_of_instruction_id}, End: {end_of_instruction_id}")
tokenizer.padding_side = 'left'

print("Loading Model...")
model = LlamaForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="balanced_low_0"
)
model.eval()

# ---------------------- 数据处理 ----------------------
# 加载 Prompt 列表
prompt_df = pd.read_csv(PROMPT_TRAIN_CSV)
prompt_list = prompt_df['prompt'].to_list()
prompt_list = sorted(prompt_list, key=len, reverse=True)
prompt_list_len = len(prompt_list)
print(f"Loaded {prompt_list_len} prompt templates.")

# 加载训练数据
MyDataset = glob.glob(DATA_TRAIN_GLOB)
MyDataset = [pd.read_csv(file, encoding='utf-8', header=None, names=['instruction', 'rephrase_method', 'label_true'], skiprows=1) for file in MyDataset]
data_df = pd.concat(MyDataset, ignore_index=True)
data_df = data_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
data_df.reset_index(inplace=True)
data_df.rename(columns={'index': 'index_'}, inplace=True)

# 应用模板函数
rows_per_template = len(data_df) // prompt_list_len

def apply_template(row):
    template_index = (row.index_ // rows_per_template) % prompt_list_len
    template = prompt_list[template_index]
    # 使用配置的特殊Token
    wrapped_inst = f' {SPECIAL_TOKENS["begin"]} {row["instruction"]} {SPECIAL_TOKENS["end"]} '
    row['instruction'] = template.format(instruction=wrapped_inst)
    return row

data_df = data_df.apply(apply_template, axis=1)

# ---------------------- Dataset & Dataloader ----------------------
class RewrittenDataset(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df
        print(data_df.head())

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        data_col = self.data_df.iloc[index, :]
        return data_col.instruction, data_col.rephrase_method, data_col.label_true
    
    def collate_fn(self, batch):
        batch_conversation = []
        batch_instruction = []
        batch_label_true = []
        batch_rephrase_method = []
        
        for instruction, rephrase_method, label_true in batch:
            conv = get_conv_template("llama-2")
            conv.append_message(conv.roles[0], instruction)
            conv.append_message(conv.roles[1], None)
            conversation = conv.get_prompt()
            
            batch_conversation.append(conversation)
            batch_instruction.append(instruction)
            batch_label_true.append(label_true)
            batch_rephrase_method.append(rephrase_method)
        
        return {
            'conversations': batch_conversation, 
            'instructions': batch_instruction, 
            'labels_true': batch_label_true, 
            'rephrase_method': batch_rephrase_method
        }

def get_loader(data_df, batch_size, shuffle=False, num_workers=4):
    ds_df = RewrittenDataset(data_df)
    dataloader_class = partial(DataLoader, pin_memory=True)
    loader = dataloader_class(
        ds_df, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=ds_df.collate_fn,
        num_workers=num_workers
    )
    return loader

# ---------------------- 特征提取逻辑 ----------------------
def make_instruction_attention_mask(input_ids, attention_masks):
    batch_size, sequence_length = input_ids.size()
    new_input_ids = torch.zeros((batch_size, sequence_length-2), dtype=torch.long, device=input_ids.device)
    new_attention_masks = torch.zeros((batch_size, sequence_length-2), dtype=torch.long, device=input_ids.device)

    attention_masks_instruction = torch.zeros(new_attention_masks.size(), device=attention_masks.device)
    end_indices = []
    start_indices = []
    
    for i, (input_id, attention_mask) in enumerate(zip(input_ids, attention_masks)):
        indices_where = torch.where((input_id == begin_of_instruction_id) | (input_id == end_of_instruction_id))[0]
        start = indices_where[0]
        end = indices_where[1].item() - 3 
        
        attention_masks_instruction[i, start:end] = 1
        end_indices.append(end)
        start_indices.append(start)
        
        mask = (input_id != begin_of_instruction_id) & (input_id != end_of_instruction_id)
        new_input_ids[i] = input_id[mask]
        new_attention_masks[i] = attention_mask[mask]

    return new_input_ids, new_attention_masks, attention_masks_instruction, end_indices, start_indices

def batch_infer(model, dataloader, save_path):
    train_label_list = []
    # 使用配置的层数
    attention_feature_list = [[] for _ in range(NUM_HIDDEN_LAYERS)]

    for batch in tqdm(dataloader):
        input_ = tokenizer(batch['conversations'], return_tensors="pt", padding=True).to('cuda')
        input_ids = input_['input_ids']
        attention_mask = input_['attention_mask']
        
        input_ids, attention_mask, attention_masks_instruction, end_indices, start_indices = \
            make_instruction_attention_mask(input_ids, attention_mask)
        
        # 注意：这里调用了自定义的 generate 方法 (含 attention_mask_instruction 参数)
        # 请确保你的 transformers 库代码或模型代码支持此参数
        all_all_layer_hidden_states, insturction_hidden_states_all_layer = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=2, 
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
            attention_mask_instruction=attention_masks_instruction
        )
        
        batch_labels_true = batch['labels_true']
        train_label_list.extend(batch_labels_true)
        
        # 处理 Attention 特征
        for layer in range(NUM_HIDDEN_LAYERS):
            attention_hidden_states = insturction_hidden_states_all_layer[f'layer_{layer}']
            for batch_idx, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
                feat = attention_hidden_states[batch_idx, end_idx, :].unsqueeze(0)
                feat = feat.to(dtype=torch.float32).detach().cpu()
                attention_feature_list[layer].append(feat)
        

    # 保存 Attention 特征
    for layer in range(NUM_HIDDEN_LAYERS):
        layer_features = torch.cat(attention_feature_list[layer], dim=0)
        print(f"Layer {layer+1} shape: {layer_features.shape}")
        
        df = pd.DataFrame({'tensor': [layer_features[i] for i in range(layer_features.size(0))]})
        df['label'] = train_label_list
        
        save_file = os.path.join(save_path, f'train_layer_{layer+1}.pt')
        torch.save(df, save_file)
        print(f"Saved to {save_file}")


# ---------------------- 主程序入口 ----------------------
if __name__ == "__main__":
    dataloader = get_loader(
        data_df,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )
    
    batch_infer(
        model, 
        dataloader, 
        FEATURE_SAVE_DIR,  
    )