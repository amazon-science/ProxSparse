 
model_dir=openlm-research
model_subdir=open_llama_7b_v2
lambda_=1.0
batch_size=1
ctx_len=2048
samples=400 # this is related to the final checkpoint dir name
lr=1e-04
checkpoint=$samples

DIR="${model_subdir}-en_sft_final_${samples}_lr${lr}_len${ctx_len}_batch${batch_size}_lambda${lambda_}"
echo -e "Starting learning mask with ProxSparse" 
# Mask Learning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python end-to-end/main.py --model "${model_dir}/${model_subdir}" --lambda_value $lambda_ --ctx_len $ctx_len --batch_size $batch_size --samples $samples --learning_rate $lr

# checking mask
echo -e "Finished learning, now extracting binary mask. Mask stored in proximal_* directory" 
python end-to-end/mask_op.py --model "$DIR/checkpoint-$checkpoint" 

# Apply mask
echo -e "Applying mask to the original model, evaluate on C4 PPL" 
python eval/eval_mask_ppl.py --mask "$DIR/checkpoint-$checkpoint" --model "${model_dir}/${model_subdir}" --method else --ctx_len 4096 