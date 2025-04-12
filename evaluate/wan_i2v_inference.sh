model_id="/model_zoo/Wan2.1-I2V-14B-480P"

prompt=$(cat examples/prompt.txt)
image_path="examples/i2v_input.JPG"
no_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

start_time=$(date +%s)  
echo "Starting at $(date +"%H:%M:%S")"  

# Tuning
# sequential tuning
python evaluate/wan_example.py \
    --task i2v-14B \
    --size 832*480 \
    --sample_steps 30 \
    --ckpt_dir $model_id \
    --image $image_path \
    --prompt "$prompt"\
    --base_seed 42 \
    --offload False \
    --use_spas_sage_attn \
    --l1 0.03 \
    --pv_l1 0.035 \
    --model_out_path evaluate/models_dict/Wan2.1-I2V-14B-480P_0.03_0.035.pt \
    --tune
#     --parallel_tune


# Inference
python evaluate/wan_example.py  \
    --use_spas_sage_attn \
    --model_out_path evaluate/models_dict/Wan2.1-I2V-14B-480P_0.06_0.07.pt \
    --compile \
    --task i2v-14B \
    --size 832*480 \
    --sample_steps 30 \
    --ckpt_dir $model_id \
    --image $image_path \
    --prompt "$prompt"\
    --base_seed 42 \
    --offload False

end_time=$(date +%s)  
elapsed=$((end_time - start_time))  
echo "Completed at $(date +"%H:%M:%S")"  
echo "Total execution time: $elapsed seconds ($(($elapsed / 60)) minutes and $(($elapsed % 60)) seconds)"  