from huggingface_hub import snapshot_download

repo_id = "aladinggit/proxsparse_models"

# List all folders you want to include
model_dirs = [
    "Llama-2-13b-hf-en_sft_final_reg0.5_400_len4096_batch1_lambdaone0.5_lambdatwo0.25",
    "Llama-2-7b-hf-en_sft_final_400_len4096_batch1_lambda0.25",
    "Meta-Llama-3.1-8B-en_sft_final_400_lr5e-05_len4096_batch1_lambda0.85",
    "Mistral-7B-v0.1-en_sft_final_400_lr5e-05_len4096_batch1_lambda20.0",
    "Mistral-7B-v0.3-en_sft_final_400_lr5e-05_len4096_batch1_lambda25.0",
    "Qwen2.5-14B-en_sft_final_400_lr0.0001_len4096_batch1_lambda0.2",
    "open_llama_7b_v2-en_sft_final_400_lr0.0001_len2048_batch1_lambda1.0",
]

# Download all folders listed above
snapshot_download(
    repo_id=repo_id,
    allow_patterns=[f"{d}/*" for d in model_dirs],
    local_dir="proxsparse_models",
)
