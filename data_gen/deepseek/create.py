from openai import OpenAI
import time
import requests
import pandas as pd
import numpy as np
import ast
import tqdm
import argparse
import sys
import os
benchmark_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(benchmark_dir)
from api_key import deepseek_api
from llm_prompts import create_prompts
from data import get_df

def get_first_n_percent(text, percent=0.25):
    length = len(text)
    target_length = length * percent

    words = text.split()

    current_length = 0
    first_part_words = []

    for word in words:
        current_length += len(word) + 1
        if current_length > target_length:
            break
        first_part_words.append(word)
    first_part_text = " ".join(first_part_words)

    start_index = len(first_part_words)
    last_part_words = words[start_index:]

    last_part_text = " ".join(last_part_words)

    return first_part_text, last_part_text


def deepseek_chat(args, prompt,temperature = 0.7):
    prompt = prompt

    client = OpenAI(api_key=args.api_key, base_url=args.url)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a large language model trained by microsoft. \
            Follow the user's instructions carefully. Respond using markdown."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=temperature,
    )
    content = response.choices[0].message.content

    return content

def get_ds_create(args, context):
    text_first_part, text_last_part = get_first_n_percent(context, percent=args.percent)
    word_number = round(len(text_last_part))

    if args.dataset == 'tweets':
        prompt_template = create_prompts["tweets"]
        prompt = prompt_template.format(text_first_part=text_first_part)
    else:
        prompt_template = create_prompts[args.dataset]
        prompt = prompt_template.format(text_first_part=text_first_part, word_number=word_number)

    response_content = deepseek_chat(args, prompt)

    return response_content

def run_ds_create(args,df):
    ai_texts = []
    for _, context in tqdm.tqdm(df.iterrows(), total=len(df)):
        context = context.tolist()[0]
        ai_texts.append(get_ds_create(args, context))
    df['ai_n_percent'] = ai_texts
    return df

def run(args):
    df = get_df(args)[500:1000]

    df = run_ds_create(args, df)

    print(df.head())
    print(df.shape)
    print(df.columns)

    df.to_csv(f'./data_gen/LLM-texts/create/{args.dataset}_ds_create.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default=deepseek_api['key'])
    parser.add_argument('--url', type=str, default=deepseek_api['url'])
    parser.add_argument("--dataset", type=str, default=None, required=True,
                        choices=['xsum', 'squad', 'writingprompts', 'pubmedqa','openreview','tweets','blog'])
    parser.add_argument("--percent", type=float, default=0.25)
    args = parser.parse_args()

    run(args)