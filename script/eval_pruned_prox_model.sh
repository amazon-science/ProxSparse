# launch python prox_pruned_model_download.py to get the pruned model download first
CUDA_VISIBLE_DEVICES=0 python eval/ppl.py --model proxsparse_models/Llama-2-7b-hf-en_sft_final_400_len4096_batch1_lambda0.25 --ctx_len 4096

CUDA_VISIBLE_DEVICES=0 python eval/ppl.py --model proxsparse_models/Llama-2-13b-hf-en_sft_final_reg0.5_400_len4096_batch1_lambdaone0.5_lambdatwo0.25 --ctx_len 4096

CUDA_VISIBLE_DEVICES=0 python eval/ppl.py --model proxsparse_models/Qwen2.5-14B-en_sft_final_400_lr0.0001_len4096_batch1_lambda0.2 --ctx_len 4096

CUDA_VISIBLE_DEVICES=0 python eval/ppl.py --model proxsparse_models/Meta-Llama-3.1-8B-en_sft_final_400_lr5e-05_len4096_batch1_lambda0.85 --ctx_len 4096

CUDA_VISIBLE_DEVICES=0 python eval/ppl.py --model proxsparse_models/Mistral-7B-v0.1-en_sft_final_400_lr5e-05_len4096_batch1_lambda20.0 --ctx_len 4096

CUDA_VISIBLE_DEVICES=0 python eval/ppl.py --model proxsparse_models/Mistral-7B-v0.3-en_sft_final_400_lr5e-05_len4096_batch1_lambda25.0 --ctx_len 4096

CUDA_VISIBLE_DEVICES=0 python eval/ppl.py --model proxsparse_models/open_llama_7b_v2-en_sft_final_400_lr0.0001_len2048_batch1_lambda1.0 --ctx_len 2048

