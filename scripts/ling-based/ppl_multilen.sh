domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "translate")
multilens=("10" "50" "100" "200" "500")
n_sample=2000
device="cuda:0"
func='ppl'

for multilen in ${multilens[@]}; do
  for domain in ${domains[@]}; do
    echo "Testing on $domain, n_sample=$n_sample"
    python detectors/ling-based/gltrppl.py --task cross-domain --dataset $domain \
    --classifier ./detectors/ling-based/classifier/cross-domain/ppl_$domain\_n${n_sample}.pkl --multilen $multilen \
    --n_sample $n_sample --device $device --mode test --func $func
  done
done