domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
epochs=4
threshold="0.0"
n_sample=2000
multilens=("10" "50" "100" "200" "500")
device='cuda:0'

for multilen in ${multilens[@]}; do
  echo "Multilen: $multilen"
  for domain in ${domains[@]}; do
    echo "Testing domain: $domain, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
    python detectors/deberta/run_deberta.py --task cross-domain --dataset $domain --mode test \
    --deberta_model ./detectors/deberta/weights/cross-domain/deberta_$domain\_ep$epochs\_thres$threshold\_n$n_sample \
    --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen
  done
done