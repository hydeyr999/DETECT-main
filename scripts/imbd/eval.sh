domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=('create' "translate" "polish" "expand" "refine" "summary" "rewrite")
device='cuda:1'

for DATASET in ${domains[@]}; do
  echo "Processing cross-domain: $DATASET"
  python detectors/imbd/run_spo.py \
      --eval_only True \
      --from_pretrained "./detectors/imbd/ckpt/cross-domain/${DATASET}_spo_lr_0.0001_beta_0.05_a_1" \
      --eval_dataset "$DATASET" \
      --device $device \
      --task_name cross-domain
done

for MODEL in ${models[@]}; do
  echo "Processing cross-model: $MODEL"
  python detectors/imbd/run_spo.py \
      --eval_only True \
      --from_pretrained "./detectors/imbd/ckpt/cross-model/${MODEL}_spo_lr_0.0001_beta_0.05_a_1" \
      --eval_dataset "$MODEL" \
      --device $device \
      --task_name cross-model
done

for OPERATION in ${operations[@]}; do
  echo "Processing cross-operation: $OPERATION"
  python detectors/imbd/run_spo.py \
      --eval_only True \
      --from_pretrained "./detectors/imbd/ckpt/cross-operation/${OPERATION}_spo_lr_0.0001_beta_0.05_a_1" \
      --eval_dataset "$OPERATION" \
      --device $device \
      --task_name cross-operation
done
