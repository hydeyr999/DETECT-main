domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
epochs=1
n_sample=2000
multilens=("10" "50" "100" "200" "500")
device='cuda:1'

for multilen in "${multilens[@]}"; do
  echo "Testing multilens: $multilen"
  for domain in ${domains[@]}; do
    echo "Testing domain: $domain, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
    python detectors/DPIC/run_dpic.py --task cross-domain --dataset $domain --mode test \
    --dpic_ckpt ./detectors/DPIC/weights/cross-domain/dpic_$domain\_ep$epochs\_n$n_sample.pth \
    --device $device --epochs $epochs --n_sample $n_sample --multilen $multilen
  done
done