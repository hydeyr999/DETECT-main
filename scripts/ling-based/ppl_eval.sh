domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "translate")
multilen=0
n_sample=2000
device="cuda:0"
func='ppl'

for domain in ${domains[@]}; do
  echo "Testing on $domain, n_sample=$n_sample"
  python detectors/ling-based/gltrppl.py --task cross-domain --dataset $domain \
  --classifier ./detectors/ling-based/classifier/cross-domain/ppl_$domain\_n${n_sample}.pkl --multilen $multilen \
  --n_sample $n_sample --device $device --mode test --func $func
done

for model in ${models[@]}; do
  echo "Testing on $model, n_sample=$n_sample"
  python detectors/ling-based/gltrppl.py --task cross-model --dataset $model \
  --classifier ./detectors/ling-based/classifier/cross-model/ppl_$model\_n${n_sample}.pkl --multilen $multilen \
  --n_sample $n_sample --device $device --mode test --func $func
done

for operation in ${operations[@]}; do
  echo "Testing on $operation, n_sample=$n_sample"
  python detectors/ling-based/gltrppl.py --task cross-operation --dataset $operation \
  --classifier ./detectors/ling-based/classifier/cross-operation/ppl_$operation\_n${n_sample}.pkl --multilen $multilen \
  --n_sample $n_sample --device $device --mode test --func $func
done