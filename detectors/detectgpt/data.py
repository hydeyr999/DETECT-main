import pandas as pd
import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def get_df(args):
    if args.multilen > 0:
        test_dir = f'./data/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
        df = pd.read_csv(test_dir)
        df['text'] = df[f'text_{args.multilen}']
    else:
        test_dir = f'./data/{args.task}'
        path = f'{test_dir}/{args.dataset}_sample.csv'
        df = pd.read_csv(path)

    print(df.head())
    print(df.shape)

    return df

def get_model(model_path,device):
    device_map = {'' : int(device.split(':')[1])}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if model_path == 'tinyllama':
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            use_cache=False,
            device_map=device_map
        )

    return model,tokenizer