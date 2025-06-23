'''
From https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''
from transformers import AutoTokenizer, AutoModelForCausalLM


import numpy as np
import torch
from tqdm import tqdm
import random

def explicit_sparsify_magnitude_row(weight, n, m): 
    dim = weight.shape
    weight = weight.view(-1, m) 
    w_mask = torch.zeros_like(weight)
    w_mask.scatter_(1, torch.topk(torch.abs(weight), n, dim = 1, largest = True)[1], True)
    w_mask = w_mask.view(dim[0], dim[1])
    return w_mask

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

seed_everything(42)


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only',
                           'penn_treebank',
                           split='validation')

    trainenc = tokenizer("\n\n".join(traindata['sentence']),
                         return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train')
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train')
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, tokenizer)
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, tokenizer)
        return get_c4(nsamples, seed, seqlen, tokenizer)


def get_test_tokens(name, tokenizer, seed=0, seqlen=2048):
    train_samples = 0
    if name == 'wikitext2':
        return get_wikitext2(train_samples, seed, seqlen, tokenizer)[1]['input_ids']
    elif name == 'c4':
        return get_c4(train_samples, seed, seqlen, tokenizer)[1].input_ids
    elif name == 'c4_new':
        return get_c4_new(train_samples, seed, seqlen, tokenizer)[1].input_ids
    else:
        raise Exception
    
    


def get_perp(model, tokenizer, seed, seqlen, datasets=['wikitext2', 'c4']):
    results = {}
    model.eval()
    with torch.no_grad():
        for dataset in datasets:
            input_tok = get_test_tokens(dataset,
                                        tokenizer,
                                        seed=seed,
                                        seqlen=seqlen)
            nsamples = input_tok.numel() // seqlen
            input_tok = input_tok[0, :(seqlen * nsamples)].view(
                nsamples, seqlen)

            # if not args.no_use_cuda_graph:
            #     model.reset()

            loss_fct = torch.nn.CrossEntropyLoss().cuda()
            acc_loss = 0.0
            progress = tqdm(range(nsamples))
            for ii in progress:
                input = input_tok[ii, :].cuda().view(1, -1)
                output = model(input,
                            use_cache=False,
                            output_hidden_states=False,
                            output_attentions=False)[0]
                shift_logits = output[:, :-1, :].contiguous()
                shift_labels = input[:, 1:]
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1))
                acc_loss += loss.item()
                progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

            avg_loss = acc_loss / nsamples

            ppl = torch.exp(torch.tensor(avg_loss)).item()
            print(f'{dataset} perplexity: {ppl}')
            results[dataset] = ppl
        return results
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default= "meta-llama/Meta-Llama-3-8B", 
                    help='Provide the model name for finetuning')
parser.add_argument('--mask', type=str, default= "meta-llama/Meta-Llama-3-8B", 
                    help='Provide the mask name for testing')
parser.add_argument('--method', type=str, default= "proximal", 
                    help='proximal')
parser.add_argument('--ctx_len', type=int, default= "1024", 
                    help='proximal')

args = parser.parse_args()
model_name = args.model



# model_name = "meta-llama/Meta-Llama-3-8B"
model_name = args.model
method = args.method
mask_name = args.mask
ctx_len = args.ctx_len

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map= "auto",
)


mask_set = []
def loop_lambda(model):
    for n, m in model.named_parameters():
        if "bias" not in n and ("k_proj" in n or "q_proj" in n or "v_proj" in n or "o_proj" in n or "up_proj" in n or "down_proj" in n or "gate_proj" in n or "out_proj" in n or "fc1" in n or "fc2" in n):
            # import ipdb; ipdb.set_trace()
            if method == "projected":
                mask = torch.load(f"./projected_mask/{n}.pt")
                mask_set.append(mask.bool())
            if method == "proximal":
                mask = torch.load(f"./proximal_mask/{n}.pt")
                mask_set.append(mask.bool())
            if method == "else":
                mask = torch.load(f"./proximal_{mask_name}/{n}.pt")
                mask_set.append(mask.bool())
loop_lambda(model)

from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
class ProxLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, mask=None):
        super(ProxLinear, self).__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.mask = mask # inverse

    def forward(self, input):
        qw = self.weight * self.mask
        return F.linear(input, qw, self.bias)
        


    def modify_weight(self, weight, mask):
        qw = weight * mask
        return qw

    def sparsify_magnitude(self, weight, n, m): 
        dim = weight.shape
        weight = weight.view(-1, m) 
        w_mask = torch.zeros_like(weight)
        w_mask.scatter_(1, torch.topk(torch.abs(weight), n, dim = 1, largest = True)[1], True)
        weight = w_mask * weight
        weight = weight.view(dim[0], dim[1])
        return weight
    
    
def replace_linear(model):
    with torch.no_grad():
        for name, m in model.named_children():
            if isinstance(m, nn.Linear) and ("bias" not in name and ("k_proj" in name or "v_proj" in name or "q_proj" in name or "o_proj" in name
                                             or "up_proj" in name or "down_proj" in name or "gate_proj" in name or "out_proj" in name or "fc1" in name or "fc2" in name)):
                newlinear = ProxLinear(m.in_features, m.out_features, m.bias is not None, device = m.weight.device, dtype=m.weight.dtype, mask = (mask_set[0]).to(m.weight.device).to(m.weight.dtype))
                del mask_set[0]
                newlinear.weight.data.copy_(m.weight.data * newlinear.mask) 
                if m.bias is not None:
                    newlinear.bias.data.copy_(m.bias.data)
                setattr(model, name, newlinear)
            elif isinstance(m, torch.nn.LayerNorm):
                pass
            else:
                replace_linear(m)

print("# begin replacing layers")
replace_linear(model)
print("# replacing sucessful!")

def reshape_weights(weight_matrix):
    m, n = weight_matrix.shape
    weight_matrix = weight_matrix.view(-1, 4)
    return weight_matrix, m, n
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

print(f"2/4 sparsity is {two_sparse/total}, one sparse is {one_sparse/total}, dense is {dense/total},  three sparse is {three_sparse/total}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
max_tokenized_len = ctx_len
res = get_perp(model, tokenizer, 42, ctx_len, ["wikitext2"])
