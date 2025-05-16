# DETECT-main
This program is the code of 'DETECT: A Comprehensive Benchmark for Detecting LLM-generated Text'.

## Requirements

Python dependencies:

    conda create -n aigtd python=3.10
    conda activate aigtd
    pip install -r requirements.txt

## Models and Datas

Next, here are the models required in this program, you can choose to download them or use them online. If you choose to download these models, please create a file named 'models' in the root.

1.DeBERTa-v3-large

2.DeBERTa-v3-small

3.gpt2-medium

4.gpt-neo-2.7B

5.llama2-7b

6.llama3.1-8b-v2

7.Mistral-7B-v0.1

8.t5-base

9.tinyllama

10.xlnet-large

After that, please download some necessary datas to run the code, the google drive links are below:

1.data: https://drive.google.com/drive/folders/1YroFpkJaam4G6RfnA4WRggce5hGePMzT?usp=drive_link

2.dpic_data: https://drive.google.com/drive/folders/1STP_d2Umw_7pq8flt9Tr4Wx5HU5b250L?usp=drive_link

3.logprobs: https://drive.google.com/drive/folders/1huPsFa5gu73VH9eogS4dLqpUIiyaOVtB?usp=drive_link

4.lingfeatures: https://drive.google.com/drive/folders/1iXy7Eg_nbD2wox0dpu5-Dany0xoJOOeM?usp=drive_link

5.human: https://drive.google.com/drive/folders/1ijmevGIWfQMKUHZQO_l6-04Zt1uALmt6?usp=drive_link

Please place the files strictly in accordance with the following project structure to ensure that the program can read the path correctly:

```html
📂 Project Structure

```text
.
├── data/
├── data_gen/
│   └── human/
├── detectors/
│   ├── ling-based/
│   │   ├── logprobs/
│   │   └── lingfeatures/
│   └── DPIC/
├── models/
│   ├── deberta-v3-large/
│   ├── llama2-7b/
│   └── .../
└── scripts/
```
</details>
```


## Train and Evaluation

If you want to reproduct our experiment results, you can feel free to run the scripts. Here are some examples:


    bash scripts/mistral/train.sh
    bash scripts/mistral/eval.sh
    bash scripts/mistral/multilen.sh
    bash scripts/mistral/imbd.sh

.
Note: Part of our codes draw from the detectors we evaluated in the experiments, and we also refer to https://github.com/rbiswasfc/llm-detect-ai.


