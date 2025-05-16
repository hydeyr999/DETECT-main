domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
multilens=("10" "50" "100" "200" "500")
device='cuda:0'

for multilen in ${multilens[@]}; do
  echo "Running multilens $multilen"
  for domain in ${domains[@]}; do
    echo "Evaluating domain: $domain"
    python detectors/baselines/baselines.py --task cross-domain --dataset $domain --device $device --multilen $multilen
  done
done