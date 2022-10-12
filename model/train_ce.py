from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)
import random
import json
from accelerate import Accelerator
import numpy as np
import torch
import torch.nn as nn
import logging
from datasets import load_dataset
import argparse
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(random_seed):
    """
    Random number fixed
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)

def preprocess_function_aligned_data(datapoint):
    """
    Specific preprocessing function for the aligned dataset
    args:
        datapoint (dict) : A datapoint dict would contain all the correponding dicts.
    """
    prompt_string = ""
    for key in datapoint:
        prompt_string += "[" + key + "]"
        prompt_string += datapoint[key]
    return prompt_string

def preprocess_function_unaligned_data(datapoint):
    pass

class CEDataset(Dataset):
    def __init__(self,tokenizer,dataset_path) -> None:
        self.dataset = json.load(open(dataset_path,"r"))
        logging.info(f"Succesfully loaded the dataset from {dataset_path} of length {len(self.dataset)}")
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        datapoint = self.dataset[idx]
        input,output = preprocess_function_aligned_data(datapoint)
        return input,output



class CrossEncoder(nn.Module):
    def __init__(self,model_name:str="microsoft/codebert-base") -> None:
        super().__init__()
        self.config  = AutoConfig.from_pretrained(model_name) 
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.classifier = torch.nn.Linear(self.config.hidden_size,1)

    def forward(self,input_ids, attention_mask,token_type_ids):
        outputs = self.encoder(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits



def train(args):


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)    
    ce_model = CrossEncoder(args.model_name)
    logger.info(f"Successfully loaded the model to memory")

    ce_dataset = CEDataset(tokenizer=tokenizer,dataset_path=args.dataset_file_path)
    ce_dataloader = DataLoader(ce_dataset,args.batch_size,shuffle = True)
    
    #
    
    #Accelerate Init
    accelerator = Accelerator()
    device = accelerator.device





    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__name__)
    
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--dataset_file_path",type=str,default="dataset/aligned_data_with_keys.json")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--output_path",type=str,default="./output")
    parser.add_argument("--batch_size",type=int,default=4)


    args = parser.parse_args()
    
    #TODO(reshinth) : Set seed function
    set_seed(args.random_seed)
    train(args)
    


