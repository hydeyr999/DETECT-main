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
from api_key import gpt_api
from llm_prompts import translate_prompts
from data import get_df

def gpt4o_chat(args,prompt,temperature=0.7):
    client = OpenAI(base_url=args.url,
                    api_key=args.api_key)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a large language model.\
            Follow the user's instructions carefully. Respond using markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    content = completion.choices[0].message.content

    return content

def get_4o_translate(args, context):
    try:
        prompt_template = translate_prompts['trans_to_ch']
        prompt = prompt_template.format(context=context)
        response_content = gpt4o_chat(args, prompt)

        prompt_template = translate_prompts['trans_to_en']
        prompt = prompt_template.format(context=response_content)
        response_content = gpt4o_chat(args, prompt)
    except Exception as e:
        response_content = None

    return response_content

def run_4o_translate(args,df):
    ai_texts = []
    for _, context in tqdm.tqdm(df.iterrows(), total=len(df)):
        context = context.tolist()[0]
        ai_texts.append(get_4o_translate(args, context))
    df['ai'] = ai_texts
    return df

def run(args):
    df = get_df(args)[1000:1500]

    df = run_4o_translate(args, df)

    print(df.head())
    print(df.shape)
    print(df.columns)

    df.to_csv(f'./data_gen/LLM-texts/translate/{args.dataset}_4o_translate_new.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default=gpt_api['url'])
    parser.add_argument("--url", type=str, default=gpt_api['key'])
    parser.add_argument("--dataset", type=str, default=None, required=True,
                        choices=['xsum','squad','writingprompts','pubmedqa','openreview','tweets','blog'])
    args = parser.parse_args()

    run(args)