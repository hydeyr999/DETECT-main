domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "translate")
multilen=0
device='cuda:0'

for domain in ${domains[@]}; do
  echo "Evaluating domain: $domain"
  python detectors/detectgpt/detectgpt.py --task cross-domain --dataset $domain --device $device --multilen $multilen
done

for model in ${models[@]}; do
  echo "Evaluating model: $model"
  python detectors/detectgpt/detectgpt.py --task cross-model --dataset $model --device $device --multilen $multilen
done

for operation in ${operations[@]}; do
  echo "Evaluating operation: $operation"
  python detectors/detectgpt/detectgpt.py --task cross-operation --dataset $operation --device $device --multilen $multilen
done