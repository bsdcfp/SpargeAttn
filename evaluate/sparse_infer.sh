
#!/bin/bash

print_help() {
  echo "Usage: $0 [full|tune|sparse] [--l1 val] [--pv_l1 val] [--out path] [--parallel_tune] [--config path]"
  echo ""
  echo "Modes:"
  echo "  full                   Inference by full attention"
  echo "  tune                   Train/tune by sparse attention"
  echo "  sparse                 Inference by sparse attention"
  echo ""
  echo "Optional arguments:"
  echo "  --l1 VAL               Set l1 parameter (overrides config)"
  echo "  --pv_l1 VAL            Set pv_l1 parameter (overrides config)"
  echo "  --out PATH             Output model path (overrides config)"
  echo "  --parallel_tune        Enable parallel tune (for tune mode)"
  echo "  --config PATH          Config file path (default: ./config.json)"
  echo "  -h, --help             Show this help message"
}

# check for no arguments or help
if [[ $# -lt 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  print_help
  exit 0
fi

mode=$1
shift

# 默认配置文件路径
config_path="./config.json"

# 默认参数（备用，通常由 config 提供）
l1=""
pv_l1=""
model_out_path=""
parallel_tune_flag=""

# 先解析 --config
extra_args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      config_path="$2"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

# 读取 config.json 参数（直接用 py_file 字段）  
if [ -f "$config_path" ]; then  
  py_file=$(jq -r '.py_file // empty' "$config_path")  
  l1_cfg=$(jq -r '.l1 // empty' "$config_path")  
  pv_l1_cfg=$(jq -r '.pv_l1 // empty' "$config_path")  
  model_out_path_cfg=$(jq -r '.model_out_path // empty' "$config_path")  
else  
  echo "Warning: Config file $config_path not found, using defaults/cli args"  
fi  

# 使用 config 里的值
l1="${l1_cfg:-$l1}"
pv_l1="${pv_l1_cfg:-$pv_l1}"
model_out_path="${model_out_path_cfg:-$model_out_path}"

# 重新解析 extra_args 覆盖参数
set -- "${extra_args[@]}"
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

# 检查参数是否完整
if [[ -z "$py_file" ]]; then
  echo "Error: 'py_file' field not found in config file or not set"
  exit 1
fi

if [[ -z "$l1" ]]; then
  echo "Error: l1 not set (neither from config nor command line)"
  exit 1
fi

if [[ -z "$pv_l1" ]]; then
  echo "Error: pv_l1 not set (neither from config nor command line)"
  exit 1
fi

if [[ -z "$model_out_path" ]]; then
  echo "Error: model_out_path not set (neither from config nor command line)"
  exit 1
fi

start_time=$(date +%s)
echo "Starting at $(date +"%H:%M:%S")"
if [[ $mode == "full" ]]; then
  echo "Running inference by full attention..."
  python "$py_file"

elif [[ $mode == "tune" ]]; then
  echo "Running sequential tuning with l1=${l1}, pv_l1=${pv_l1}, parallel_tune:${parallel_tune_flag}..."
  python "$py_file" \
      --use_spas_sage_attn \
      --l1 $l1 \
      --pv_l1 $pv_l1 \
      --model_out_path $model_out_path \
      --tune \
      $parallel_tune_flag

elif [[ $mode == "sparse" ]]; then
  echo "Running inference by sparse attention with l1=${l1}, pv_l1=${pv_l1}..."
  python "$py_file" \
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
