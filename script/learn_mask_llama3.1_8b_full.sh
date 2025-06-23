model_dir=meta-llama
model_subdir=Llama-3.1-8B
lambda_=0.85
lambda2_=1.0
project_lambda2=1
epsilon=0.1
batch_size=1
ctx_len=4096
samples=400 # this is related to the final checkpoint dir name
lr=5e-05
checkpoint=$samples

DIR="${model_subdir}-en_sft_final_reg${epsilon}_${samples}_lr${lr}_len${ctx_len}_batch${batch_size}_lambdaone${lambda_}_lambdatwo${lambda2_}"
echo -e "Starting learning mask with ProxSparse" 
# Mask Learning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python end-to-end/main.py --model "${model_dir}/${model_subdir}" --lambda_value $lambda_ --lambda2_value $lambda2_ --project_lambda2 $project_lambda2 --ctx_len $ctx_len --batch_size $batch_size --samples $samples  --epsilon $epsilon --learning_rate $lr

# checking mask
echo -e "Finished learning, now extracting binary mask. Mask stored in proximal_* directory" 
python end-to-end/mask_op.py --model "$DIR/checkpoint-$checkpoint" >> prune_sparsity.txt

# Apply mask
echo -e "Applying mask to the original model, evaluate on C4 PPL" 
python eval/eval_mask_ppl.py --mask "$DIR/checkpoint-$checkpoint" --model "${model_dir}/${model_subdir}" --method else --ctx_len 4096 >> prune_full_iteration.txt

