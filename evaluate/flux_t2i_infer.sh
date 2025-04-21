#!/bin/bash  

print_help() {  
  echo "Usage: $0 [full|tune|sparse] [--l1 val] [--pv_l1 val] [--out path] [--parallel_tune]"  
  echo ""  
  echo "Modes:"  
  echo "  full                   Inference by full attention"  
  echo "  tune                   Train/tune by sparse attention"  
  echo "  sparse                 Inference by sparse attention"  
  echo ""  
  echo "Optional arguments:"  
  echo "  --l1 VAL               Set l1 parameter (default: 0.03)"  
  echo "  --pv_l1 VAL            Set pv_l1 parameter (default: 0.035)"  
  echo "  --out PATH             Output model path"  
  echo "  --parallel_tune        Enable parallel tune (for tune mode)"  
  echo "  -h, --help             Show this help message"  
}  


# check for no arguments or help  
if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then  
  print_help  
  exit 0  
fi  

mode=$1  
shift  

# 默认参数  
l1=0.06 
pv_l1=0.07  
model_out_path="evaluate/models_dict/flux_saved_state_dict_${l1}_${pv_l1}.pt"  
parallel_tune_flag=""  

# 解析参数  
while [[ $# -gt 0 ]]; do  
  case $1 in  
    --l1)  
      l1="$2"  
      shift 2  
      ;;  
    --pv_l1)  
      pv_l1="$2"  
      shift 2  
      ;;  
    --out)  
      model_out_path="$2"  
      shift 2  
      ;;  
    --parallel_tune)  
      parallel_tune_flag="--parallel_tune"  
      shift  
      ;;  
    -h|--help)  
      print_help  
      exit 0  
      ;;  
    *)  
      echo "Error: Unknown parameter: $1"  
      print_help  
      exit 1  
      ;;  
  esac  
done  

start_time=$(date +%s)  
echo "Starting at $(date +"%H:%M:%S")"  
if [[ $mode == "full" ]]; then  
  echo "Running inference by full attention..."  
  python evaluate/flux_example.py  

elif [[ $mode == "tune" ]]; then  
  echo "Running sequential tuning with l1=${l1}, pv_l1=${pv_l1}, parallel_tune:${parallel_tune_flag}..."  
  python evaluate/flux_example.py \
      --use_spas_sage_attn \
      --l1 $l1 \
      --pv_l1 $pv_l1 \
      --model_out_path $model_out_path \
      --tune \
      $parallel_tune_flag  

elif [[ $mode == "sparse" ]]; then  
  echo "Running inference by sparse attention with l1=${l1}, pv_l1=${pv_l1}..."  
  python evaluate/flux_example.py \
      --use_spas_sage_attn \
      --l1 $l1 \
      --pv_l1 $pv_l1 \
      --model_out_path $model_out_path  

else  
  echo "Invalid mode: $mode"  
  print_help  
  exit 1  
fi  

end_time=$(date +%s)  
elapsed=$((end_time - start_time))  
echo "Completed at $(date +"%H:%M:%S")"  
echo "Total execution time: $elapsed seconds ($(($elapsed / 60)) minutes and $(($elapsed % 60)) seconds)"  
