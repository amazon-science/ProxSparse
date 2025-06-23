import evaluate

import datasets
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import os

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
perplexity = evaluate.load("perplexity", module_type="metric") # previously 0
input_texts = datasets.load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train")["text"][:3000]


def explicit_sparsify_magnitude_row(weight, n, m): # prune and leave n out of m in 2d array
    dim = weight.shape
    weight = weight.view(-1, m) # for a 2d array, reshale into 4 blocks
    w_mask = torch.zeros_like(weight)
    w_mask.scatter_(1, torch.topk(torch.abs(weight), n, dim = 1, largest = True)[1], True)
    w_mask = w_mask.view(dim[0], dim[1])
    return w_mask

def reshape_weights(weight_matrix):
    m, n = weight_matrix.shape
    weight_matrix = weight_matrix.view(-1, 4)
    return weight_matrix, m, n


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default= "meta-llama/Meta-Llama-3-8B", 
                    help='Provide the model name for finetuning')
parser.add_argument('--method', type=str, default= "proximal", 
                    help='proximal')

args = parser.parse_args()
model_name = args.model
method = args.method

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map= "auto",
)
one_sparse = 0 
two_sparse = 0
three_sparse = 0
dense = 0
total = 0

for n, m in model.named_parameters():
    if "bias" not in n and ("k_proj" in n or "q_proj" in n or "v_proj" in n or "o_proj" in n or "up_proj" in n or "down_proj" in n or "gate_proj" in n or "out_proj" in n or "fc1" in n or "fc2" in n):
        mm, _, __ = reshape_weights(m)
        total += mm.shape[0]
        zero_counts = (mm == 0).sum(dim=1)
        num_zeros_0 = (zero_counts == 0).sum().item()
        num_zeros_1 = (zero_counts == 1).sum().item()
        num_zeros_2 = (zero_counts == 2).sum().item()
        num_zeros_3 = (zero_counts == 3).sum().item()

        dense += num_zeros_0
        one_sparse += num_zeros_1
        two_sparse += num_zeros_2
        three_sparse += num_zeros_3

        mask = explicit_sparsify_magnitude_row(m, 2, 4)
        if not os.path.isdir(f"./proximal_{model_name.split('/')[0]}"):
            os.mkdir(f"./proximal_{model_name.split('/')[0]}")
        if not os.path.isdir(f"./proximal_{model_name}"):
            os.mkdir(f"./proximal_{model_name}")
        torch.save(mask, f"./proximal_{model_name}/{n}.pt")

print("one sparse ratio: ", one_sparse/total)
print("two sparse ratio: ", two_sparse/total)
print("three sparse ratio: ", three_sparse/total)
print("dense ratio: ", dense/total)
