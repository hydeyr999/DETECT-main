domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "translate")
epochs=4
n_sample=2000
device='cuda:1'

for domain in ${domains[@]}; do
  echo "Training on $domain, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
  python detectors/DPIC/run_dpic.py --task cross-domain --dataset $domain --mode train \
  --device $device --epochs $epochs --n_sample $n_sample
done

for model in ${models[@]}; do
  echo "Training on $model, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
  python detectors/DPIC/run_dpic.py --task cross-model --dataset $model --mode train \
  --device $device --epochs $epochs --n_sample $n_sample
done

for operation in ${operations[@]}; do
  echo "Training on $operation, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
  python detectors/DPIC/run_dpic.py --task cross-operation --dataset $operation --mode train \
  --device $device --epochs $epochs --n_sample $n_sample
done