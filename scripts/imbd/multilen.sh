domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
cutlens=("10" "50" "100" "200" "500")
device='cuda:1'

for CUTLEN in ${cutlens[@]}; do
  for DATASET in ${domains[@]}; do
    echo "Processing cross-domain: $DATASET"
    python detectors/imbd/run_spo.py  \
        --eval_only True \
        --from_pretrained "./detectors/imbd/ckpt/cross-domain/${DATASET}_spo_lr_0.0001_beta_0.05_a_1" \
        --eval_dataset "$DATASET" \
        --device $device \
        --task_name cross-domain \
        --cut_len "$CUTLEN"
  done
done