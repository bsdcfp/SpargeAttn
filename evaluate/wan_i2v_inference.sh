# TFP Values: 
# Set the following values to control the percentage of timesteps using dense attention:
# 35% → 0.09, 30% → 0.075, 25% → 0.055, 20% → 0.045, 15% → 0.03, 10% → 0.02
first_times_fp=0.075
first_layers_fp=0.025
sparsity=0.25
pattern="SVG" # choose from [dense, SVG]
model_id="/model_zoo/Wan2.1-I2V-14B-480P-Diffusers"

prompt=$(cat examples/wan/3/prompt.txt)
image_path="examples/wan/3/image.jpg"

start_time=$(date +%s)  
echo "Starting at $(date +"%H:%M:%S")"  

cmd="python wan_i2v_inference.py \
    --model_id "$model_id" \
    --prompt "$prompt" \
    --image_path "$image_path" \
    --seed 0 \
    --num_inference_steps 30 \
    --pattern ${pattern} \
    --num_sampled_rows 64 \
    --sparsity $sparsity \
    --sample_mse_max_row 20000 \
    --first_times_fp $first_times_fp \
    --first_layers_fp $first_layers_fp"
echo $cmd

end_time=$(date +%s)  
elapsed=$((end_time - start_time))  
echo "Completed at $(date +"%H:%M:%S")"  
echo "Total execution time: $elapsed seconds ($(($elapsed / 60)) minutes and $(($elapsed % 60)) seconds)"  