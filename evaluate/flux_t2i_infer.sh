
start_time=$(date +%s)  
echo "Starting at $(date +"%H:%M:%S")"  

# Tuning
# sequential tuning
l1=0.03
pv_l1=0.035
model_out_path="evaluate/models_dict/flux_saved_state_dict_${l1}_${pv_l1}.pt"
python evaluate/flux_example.py \
    --use_spas_sage_attn \
    --l1 $l1 \
    --pv_l1 $pv_l1 \
    --model_out_path $model_out_path \
    --tune
#     --parallel_tune


# Inference
# python evaluate/flux_example.py \
#     --use_spas_sage_attn \
#     --l1 $l1 \
#     --pv_l1 $pv_l1 \
#     --model_out_path $model_out_path 

end_time=$(date +%s)  
elapsed=$((end_time - start_time))  
echo "Completed at $(date +"%H:%M:%S")"  
echo "Total execution time: $elapsed seconds ($(($elapsed / 60)) minutes and $(($elapsed % 60)) seconds)"  
