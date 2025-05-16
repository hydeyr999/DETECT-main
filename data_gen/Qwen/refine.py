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
from api_key import qwen_api
from llm_prompts import refine_prompts
from data import get_df

def qwen_chat(args,prompt,temperature=0.7):
    client = OpenAI(base_url=args.url,
                    api_key=args.api_key)

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a large language model trained by microsoft.\
            Follow the user's instructions carefully. Respond using markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    content = completion.choices[0].message.content

    return content

def get_qwen_refine(args,context):

    prompt_template = refine_prompts[args.dataset]
    prompt = prompt_template.format(context=context)

    try:
        response_content = qwen_chat(args,prompt)
    except Exception as e:
        response_content = None

    return response_content

def run_qwen_refine(args,df):
    ai_texts = []
    for _, context in tqdm.tqdm(df.iterrows(), total=len(df)):
        context = context.tolist()[0]
        ai_texts.append(get_qwen_refine(args,context))
    df['ai'] = ai_texts
    return df

def run(args):
    df = get_df(args)[2400:2500]

    df = run_qwen_refine(args,df)

    print(df.head())
    print(df.shape)
    print(df.columns)

    df.to_csv(f'./data_gen/LLM-texts/refine/{args.dataset}_Qwen_refine.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=qwen_api['key'])
    parser.add_argument("--url", type=str, default=qwen_api['url'])
    parser.add_argument("--dataset", type=str, default=None, required=True,
                        choices=['xsum', 'squad', 'writingprompts', 'pubmedqa','openreview','blog','tweets'])
    args = parser.parse_args()

    run(args)