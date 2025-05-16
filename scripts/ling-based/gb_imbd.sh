domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "translate")
n_sample=500
multilen=0
device="cuda:0"

python detectors/ling-based/ghostbuster.py --task otherdata --dataset imbd --n_sample $n_sample --device $device --mode train

for domain in ${domains[@]}; do
  echo "Testing on $domain, n_sample=$n_sample"
  python detectors/ling-based/ghostbuster.py --task cross-domain --dataset $domain \
  --mode test --classifier ./detectors/ling-based/classifier/otherdata/gb_imbd_n${n_sample}.pkl \
  --n_sample $n_sample --multilen $multilen --device $device
done

for model in ${models[@]}; do
  echo "Testing on $model, n_sample=$n_sample"
  python detectors/ling-based/ghostbuster.py --task cross-model --dataset $model \
  --mode test --classifier ./detectors/ling-based/classifier/otherdata/gb_imbd_n${n_sample}.pkl \
  --n_sample $n_sample --multilen $multilen --device $device
done

for operation in ${operations[@]}; do
  echo "Testing on $operation, n_sample=$n_sample"
  python detectors/ling-based/ghostbuster.py --task cross-operation --dataset $operation \
  --mode test --classifier ./detectors/ling-based/classifier/otherdata/gb_imbd_n${n_sample}.pkl \
  --n_sample $n_sample --multilen $multilen --device $device
done