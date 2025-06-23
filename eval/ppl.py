'''
From https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''
from transformers import AutoTokenizer, AutoModelForCausalLM


import numpy as np
import torch
from tqdm import tqdm
import random



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
        data_files={'train': 'en/c4-train.00001-of-01024.json.gz'},
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
        data_files={'train': 'en/c4-train.00001-of-01024.json.gz'},
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
parser.add_argument('--ctx_len', type=int, default= "1024", 
                    help='proximal')

args = parser.parse_args()
model_name = args.model
ctx_len = args.ctx_len

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map= "auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

max_tokenized_len = ctx_len
res = get_perp(model, tokenizer, 42, ctx_len, ["wikitext2"])
