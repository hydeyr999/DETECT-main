domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "translate")
n_sample=2000
device="cuda:0"
func='gltr'

for domain in ${domains[@]}; do
  echo "Training on $domain, n_sample=$n_sample"
  python detectors/ling-based/gltrppl.py --task cross-domain --dataset $domain \
  --n_sample $n_sample --device $device --mode train --func $func
done

for model in ${models[@]}; do
  echo "Training on $model, n_sample=$n_sample"
  python detectors/ling-based/gltrppl.py --task cross-model --dataset $model \
  --n_sample $n_sample --device $device --mode train --func $func
done

for operation in ${operations[@]}; do
  echo "Training on $operation, n_sample=$n_sample"
  python detectors/ling-based/gltrppl.py --task cross-operation --dataset $operation \
  --n_sample $n_sample --device $device --mode train --func $func
done
