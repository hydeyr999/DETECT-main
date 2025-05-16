domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "translate")
epochs=4
threshold="0.0"
n_sample=2000
multilen=0
device='cuda:0'
imbddata=True

python detectors/mistral/run_mistral.py --dataset imbd --mode train --device $device \
--epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata $imbddata

for domain in ${domains[@]}; do
  echo "Testing domain: $domain, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/mistral/run_mistral.py --task cross-domain --dataset $domain --mode test \
  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata $imbddata
done

for model in ${models[@]}; do
  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/mistral/run_mistral.py --task cross-model --dataset $model --mode test \
  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata $imbddata
done

for operation in ${operations[@]}; do
  echo "Testing operation: $operation, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/mistral/run_mistral.py --task cross-operation --dataset $operation --mode test \
  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata $imbddata
done