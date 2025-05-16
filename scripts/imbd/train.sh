domains=('xsum' "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=('create' "translate" "polish" "expand" "refine" "summary" "rewrite")
DATANUM=2000
EPOCHS=4
LR=1e-4
BETA=0.05
device='cuda:1'

# cross-domain
TASK_NAME="cross-domain"
for DATASET in "${domains[@]}"; do
  echo "Running $TASK_NAME with dataset: $DATASET"
  python detectors/imbd/run_spo.py \
    --datanum=$DATANUM \
    --task_name "$TASK_NAME" \
    --epochs=$EPOCHS \
    --lr=$LR \
    --beta=$BETA \
    --device=$device \
    --train_dataset "$DATASET" \
    --eval_dataset "$DATASET" \
    --task_name cross-domain
done

# cross-model
TASK_NAME="cross-model"
for MODEL in "${models[@]}"; do
  echo "Running $TASK_NAME with model: $MODEL"
  python detectors/imbd/run_spo.py \
    --datanum=$DATANUM \
    --task_name "$TASK_NAME" \
    --epochs=$EPOCHS \
    --lr=$LR \
    --beta=$BETA \
    --device=$device \
    --train_dataset "$MODEL" \
    --eval_dataset "$MODEL" \
    --task_name cross-model
done

# cross-operation
TASK_NAME="cross-operation"
for OP in "${operations[@]}"; do
  echo "Running $TASK_NAME with operation: $OP"
  python detectors/imbd/run_spo.py \
    --datanum=$DATANUM \
    --task_name "$TASK_NAME" \
    --epochs=$EPOCHS \
    --lr=$LR \
    --beta=$BETA \
    --device=$device \
    --train_dataset "$OP" \
    --eval_dataset "$OP" \
    --task_name cross-operation
done

