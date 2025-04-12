

start_time=$(date +%s)  
echo "Starting at $(date +"%H:%M:%S")"  

# Tuning
# sequential tuning
# python evaluate/cogvideo_example.py \
#     --use_spas_sage_attn \
#     --model_out_path evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt \
#     --tune \
#     --parallel_tune

# Inference
python evaluate/cogvideo_example.py  \
    --use_spas_sage_attn \
    --model_out_path evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt \
    --compile

end_time=$(date +%s)  
elapsed=$((end_time - start_time))  
echo "Completed at $(date +"%H:%M:%S")"  
echo "Total execution time: $elapsed seconds ($(($elapsed / 60)) minutes and $(($elapsed % 60)) seconds)"  