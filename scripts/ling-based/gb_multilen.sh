domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
n_sample=2000
multilens=("10" "50" "100" "200" "500")
device='cuda:0'

for multilens in ${multilens[@]}; do
  for domain in ${domains[@]}; do
    echo "Testing on $domain, n_sample=$n_sample"
    python detectors/ling-based/ghostbuster.py --task cross-domain --dataset $domain \
    --mode test --classifier ./detectors/ling-based/classifier/cross-domain/gb_$domain\_n${n_sample}.pkl \
    --n_sample $n_sample --multilen $multilen --device $device
  done
done