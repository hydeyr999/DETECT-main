domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "translate")
epochs=4
n_sample=2000
multilen=0
device='cuda:0'

for domain in ${domains[@]}; do
  echo "Testing domain: $domain, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/DPIC/run_dpic.py --task cross-domain --dataset $domain --mode test \
  --dpic_ckpt ./detectors/DPIC/weights/cross-domain/dpic_$domain\_ep$epochs\_n$n_sample.pth \
  --device $device --epochs $epochs --n_sample $n_sample --multilen $multilen
done

for model in ${models[@]}; do
  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/DPIC/run_dpic.py --task cross-model --dataset $domain --mode test \
  --dpic_ckpt ./detectors/DPIC/weights/cross-model/dpic_$model\_ep$epochs\_n$n_sample.pth \
  --device $device --epochs $epochs --n_sample $n_sample --multilen $multilen
done

for operation in ${operations[@]}; do
  echo "Testing operation: $operation, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/DPIC/run_dpic.py --task cross-operation --dataset $domain --mode test \
  --dpic_ckpt ./detectors/DPIC/weights/cross-operation/dpic_$operation\_ep$epochs\_n$n_sample.pth \
  --device $device --epochs $epochs --n_sample $n_sample --multilen $multilen
done