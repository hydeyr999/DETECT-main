domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=('create' "translate" "polish" "expand" "refine" "summary" "rewrite")
DATANUM=500
EPOCHS=4
LR=1e-4
BETA=0.05
device='cuda:1'

python detectors/imbd/run_spo.py \
    --datanum=$DATANUM \
    --epochs=$EPOCHS \
    --lr=$LR \
    --beta=$BETA \
    --device=$device \
    --train_dataset imbd \
    --eval_dataset xsum \

for DATASET in ${domains[@]}; do
  echo "Processing cross-domain: $DATASET"
  python detectors/imbd/run_spo.py \
      --eval_only True \
      --from_pretrained=./detectors/imbd/ckpt/ai_detection_500/imbd_spo_lr_0.0001_beta_0.05_a_1 \
      --eval_dataset "$DATASET" \
      --device $device
done

for MODEL in ${models[@]}; do
  echo "Processing cross-model: $MODEL"
  python detectors/imbd/run_spo.py \
    --eval_only True \
    --from_pretrained=./detectors/imbd/ckpt/ai_detection_500/imbd_spo_lr_0.0001_beta_0.05_a_1 \
    --eval_dataset $MODEL \
    --device $device
done

for OPERATION in ${operations[@]}; do
  echo "Processing cross-operation: $OPERATION"
  python detectors/imbd/run_spo.py \
    --eval_only True \
    --from_pretrained=./detectors/imbd/ckpt/ai_detection_500/imbd_spo_lr_0.0001_beta_0.05_a_1 \
    --eval_dataset $OPERATION \
    --device $device
done
