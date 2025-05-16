domains=('xsum' "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=('create' "translate" "polish" "expand" "refine" "summary" "rewrite")
device="cuda:0"

for domain in ${domains[@]}; do
  python detectors/opensource/radar.py --task cross-domain --dataset $domain --device $device
  python detectors/opensource/mpu.py --task cross-domain --dataset $domain --device $device
  python detectors/opensource/openai.py --task cross-domain --dataset $domain --device $device
done

for model in ${models[@]}; do
  python detectors/opensource/radar.py --task cross-model --model $model --device $device
  python detectors/opensource/mpu.py --task cross-model --model $model --device $device
  python detectors/opensource/openai.py --task cross-model --model $model --device $device
done

for operation in ${operations[@]}; do
  python detectors/opensource/radar.py --task cross-operation --operation $operation --device $device
  python detectors/opensource/mpu.py --task cross-operation --operation $operation --device $device
  python detectors/opensource/openai.py --task cross-operation --operation $operation --device $device
done