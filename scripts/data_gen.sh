datasets=('xsum' 'squad' 'writingprompts' 'pubmedqa' 'openreview' 'tweets' 'blog')
operations=('create' 'rewrite' 'summary' 'refine' 'polish' 'expand' 'translate')
models=('llama' 'deepseek' 'gpt4o' 'Qwen')

for dataset in ${datasets[@]}; do
  for operation in ${operations[@]}; do
    for model in ${models[@]}; do
      echo "Running ${operation} on ${dataset} with ${model}"
      python ./data_gen/${model}/${operation}.py --dataset ${dataset}
      echo "Finished ${operation} on ${dataset} with ${model}"
    done
  done
done
