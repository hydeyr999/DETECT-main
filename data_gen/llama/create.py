import pandas as pd
import numpy as np
import torch
import re
import ast
import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
import os
benchmark_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(benchmark_dir)
from llm_prompts import create_prompts
from data import get_df

def get_aux_llm(args):
    device_map = {"": 1}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map
    )

    return tokenizer, model

def get_first_n_percent(text, percent=0.25):
    # 计算 25% 的位置
    length = len(text)
    target_length = length * percent

    # 分割文本为单词列表
    words = text.split()

    # 计算字数总和，直到达到目标长度
    current_length = 0
    first_part_words = []

    for word in words:
        current_length += len(word) + 1  # 加 1 是因为要包括空格
        if current_length > target_length:
            break
        first_part_words.append(word)
    # 将结果单词列表重新组合成文本
    first_part_text = " ".join(first_part_words)

    start_index = len(first_part_words)  # 后 75% 的起始位置
    last_part_words = words[start_index:]

    # 将后 75% 部分的单词列表重新组合成文本
    last_part_text = " ".join(last_part_words)

    return first_part_text, last_part_text

def base_prompt_template() -> str:
    template = """<|system|>
    You are a large language model trained by microsoft. Follow the user's instructions carefully. Respond using markdown.
    <|user|>
    {query}
    <|assistant|>
    """
    return template

def get_response(model,tokenizer,prompt,temperature = 0.7,max_tokens = 3000):

    messages = [
        {"role": "system", "content": "You are a large language model trained by microsoft. Follow the user's instructions carefully."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    attention_mask = (input_ids != tokenizer.pad_token_id).int()

    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask,
        )

    response = output_ids[0][input_ids.shape[-1]:]
    content = tokenizer.decode(response, skip_special_tokens=True)

    return content


"""
Complete: for xsum, squad, pubmedqa, writingprompts datasets.
"""
def get_complete(args,model,tokenizer,text):

    text_first_part,text_last_part = get_first_n_percent(text,percent=args.percent)
    word_number = round(len(text_last_part))

    if args.dataset == 'tweets':
        prompt_template = create_prompts["tweets"]
        prompt = prompt_template.format(text_first_part=text_first_part)
    else:
        prompt_template = create_prompts[args.dataset]
        prompt = prompt_template.format(text_first_part=text_first_part,word_number=word_number)

    response_content = get_response(model, tokenizer, prompt)

    return response_content

def run_complete(args,df):
    tokenizer,model = get_aux_llm(args)
    ai_n_percent = []
    for _, context in tqdm.tqdm(df.iterrows(), total=len(df)):
        context = context.tolist()[0]
        ai = get_complete(args,model, tokenizer, context)
        ai_n_percent.append(ai)
    df['ai_n_percent'] = ai_n_percent
    return df

def run(args):
    df = get_df(args)[:500]

    df = run_complete(args, df)

    print(df.head())
    print(df.shape)
    print(df.columns)

    df.to_csv(f'./data_gen/LLM-texts/create/{args.dataset}_llama_create.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['xsum', 'writingprompts', 'squad', 'pubmedqa',
                                                        'openreview','tweets','blog'])
    parser.add_argument('--model_path', type=str, default='./models/llama3-8b-v2')
    parser.add_argument('--percent', type=float,default=0.25)
    args = parser.parse_args()

    run(args)


