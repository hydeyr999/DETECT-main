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
from llm_prompts import summary_prompts
from data import get_df

def deepseek_chat(api_key, prompt,temperature = 0.7):

    client = OpenAI(api_key=api_key, base_url=args.url)
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

def get_ds_summary(args, context):
    try:
        prompt_template = summary_prompts[args.dataset]
        prompt = prompt_template.format(context=context)

        response_content = deepseek_chat(args.api_key, prompt)
    except Exception as e:
        response_content = None
    return response_content

def run_ds_summary(args,df):
    ai_texts = []
    for _, context in tqdm.tqdm(df.iterrows(), total=len(df)):
        context = context.tolist()[0]
        ai_texts.append(get_ds_summary(args, context))
    df['ai'] = ai_texts
    return df

def run(args):
    df = get_df(args)[2700:2800]

    df = run_ds_summary(args, df)

    print(df.head())
    print(df.shape)
    print(df.columns)

    df.to_csv(f'./data_gen/LLM-texts/summary/{args.dataset}_ds_summary.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=deepseek_api['key'])
    parser.add_argument("--url", type=str, default=deepseek_api['url'])
    parser.add_argument("--dataset", type=str, default=None, required=True,
                        choices=['xsum','squad','writingprompts','pubmedqa','openreview','tweets','blog'])
    args = parser.parse_args()

    run(args)