from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch
import numpy as np
from datasets import load_dataset, DatasetDict
import random
import argparse
from trl import SFTConfig, SFTTrainer 

import config # configurations for lambdas
from patch_transformers_trainer import prox_inner_training_loop, prox_compute_loss
from prox_linear import replace_prox_linear

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default= "meta-llama/Llama-2-7b", 
                    help='Provide the model name for finetuning')
parser.add_argument('--dataset', type=str, default="en", 
                    help='Calibration dataset')
parser.add_argument('--lambda_value', type=float, default=0.1, 
                    help='lambda value')
parser.add_argument('--lambda2_value', type=float, default=0.1, 
                    help='lambda2 value')
parser.add_argument('--ctx_len', type=int, default="1024", 
                    help='ctx length ratio')
parser.add_argument('--samples', type=int, default="3200", 
                    help='samples nr')
parser.add_argument('--batch_size', type=int, default="1", 
                    help='batch size')
parser.add_argument('--project_lambda2', type=int, default="0", 
                    help='projected descent')
parser.add_argument('--epsilon', type=float, default="0.1", 
                    help='lambda2 epsilon')
parser.add_argument('--learning_rate', type=float, default=1e-4, 
                    help='lr')
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

args = parser.parse_args()
model_name = args.model
dataset_name = args.dataset
lr = args.learning_rate
dataset = load_dataset("allenai/c4", data_files="en/c4-train.00000-of-01024.json.gz")
ctx_len = args.ctx_len
samples = args.samples
batch_size = args.batch_size
epsilon = args.epsilon

config.lambda_ = args.lambda_value
config.epsilon = epsilon
config.lambda2_ = args.lambda2_value
config.project_lambda2 = args.project_lambda2

train_testvalid = dataset["train"].train_test_split(test_size=0.95, seed=42) # train 3563 on 0.99 # when data size is large, there might be memory outage, mostly 95
valid_test = train_testvalid["test"].train_test_split(test_size=0.999, seed=42) # valid 352
dataset = DatasetDict({
    'train': train_testvalid['train'].select(range(samples)), # 5x data, but 16x, in case of huggingface bug
    'validation': valid_test['train'].select(range(32)), # we prune at every checkpoint
    'test': valid_test['test']})



del train_testvalid, valid_test, dataset["test"]



model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # float 32
    device_map= "auto",
)

if config.project_lambda2 == 1:
    print("# begin replacing layers")
    replace_prox_linear(model)
    print("# replacing sucessful!")
model.train()


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "left"





dataset_id = dataset_name
BASE_MODEL = model_name

if config.project_lambda2 == 1:
    repository_id = f"{BASE_MODEL.split('/')[1]}-{dataset_id}_sft_final_reg{regularizer}_{samples}_lr{lr}_len{ctx_len}_batch{batch_size}_lambdaone{config.lambda_}_lambdatwo{config.lambda2_}"
else:
    repository_id = f"{BASE_MODEL.split('/')[1]}-{dataset_id}_sft_final_{samples}_lr{lr}_len{ctx_len}_batch{batch_size}_lambda{config.lambda_}"

sft_config = SFTConfig(
    dataset_text_field="text",
    output_dir=repository_id,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    max_seq_length=ctx_len,
    learning_rate=lr,
    num_train_epochs=1,
    optim="adamw_torch",
    warmup_ratio = 0.1, 
    logging_dir=f"{repository_id}/logs",
    logging_strategy="steps",
    logging_steps=0.1,
    logging_first_step=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps= 0.1,
    save_total_limit=10,
    load_best_model_at_end=True,
    eval_accumulation_steps=2,
    eval_steps=0.1,
    save_only_model= True,
)

# patching with Proxsparse operator
print("Patching inner training loop and loss computation")
SFTTrainer._inner_training_loop = prox_inner_training_loop
SFTTrainer.compute_loss = prox_compute_loss

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=sft_config,
)


trainer.train()
print("Finished!")