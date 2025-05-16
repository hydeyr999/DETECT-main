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
from llm_prompts import expand_prompts
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

def get_expand(model, tokenizer, context):
    prompt_template = expand_prompts[args.dataset]
    prompt = prompt_template.format(context=context)
    response_content = get_response(model, tokenizer, prompt)
    return response_content

def run_expand(args,df):
    tokenizer, model = get_aux_llm(args)
    ai_texts = []
    for _, context in tqdm.tqdm(df.iterrows(), total=len(df)):
        context = context.tolist()[0]
        ai_texts.append(get_expand(model, tokenizer, context))
    df['ai'] = ai_texts
    return df

def run(args):
    df = get_df(args)[300:400]

    df = run_expand(args, df)
    print(df.head())
    print(df.shape)
    print(df.columns)

    df.to_csv(f'./data_gen/LLM-texts/expand/{args.dataset}_llama_expand.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['xsum', 'writingprompts', 'squad', 'pubmedqa',
                                                        'openreview','tweets','blog'])
    parser.add_argument('--model_path', type=str, default='./models/llama3-8b-v2')
    args = parser.parse_args()


    run(args)
