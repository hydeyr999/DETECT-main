import argparse
import os

import pandas as pd
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from peft import PeftModel

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

from mistral.r_detect.ai_dataset import AiDataset
from mistral.r_detect.ai_loader import AiCollator, show_batch
from mistral.r_detect.ai_model import MistralForDetectAI,LlamaForDetectAI

import re
import unicodedata

# pre-process -----
char_to_remove = ['{', '£', '\x97', '¹', 'å', '\\', '\x85', '<', '\x99', \
                  'é', ']', '+', 'Ö', '\xa0', '>', '|', '\x80', '~', '©', \
                  '/', '\x93', '$', 'Ó', '²', '^', ';', '`', 'á', '*', '(', \
                  '¶', '®', '[', '\x94', '\x91', '#', '-', 'ó', ')', '}', '=']

os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda:0"
def preprocess_text(text, strategy='light'):
    assert strategy in ["none", "light", "heavy"], "pre-processing strategy must one of: none, light, heavy"

    if strategy == "none":
        text = text

    elif strategy == "light":
        text = text.encode("ascii", "ignore").decode('ascii')
        text = text.strip()
        text = text.strip("\"")

        for c in char_to_remove:
            text = text.replace(c, "")
        if len(text) == 0:
            pass
        else:
            if text[-1] != ".":
                text = text.split(".")
                text = ".".join(text[:-1])
                text += "."
    else:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s.,;?!:()\'\"%-]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

    return text


def run_inference(accelerator, model, infer_dl, example_ids):
    model.eval()
    all_predictions = []

    progress_bar = tqdm(range(len(infer_dl)), disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(infer_dl):
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits.reshape(-1)
        predictions = torch.sigmoid(logits)
        predictions = accelerator.gather_for_metrics(predictions)
        predictions = predictions.cpu().numpy().tolist()

        all_predictions.extend(predictions)

        progress_bar.update(1)
    progress_bar.close()

    # print(all_predictions)
    result_df = pd.DataFrame()
    result_df["id"] = example_ids
    result_df["generated"] = all_predictions

    return result_df


def main(cfg, save_dir, model_id,df,device):
    os.environ["ACCELERATE_TORCH_DEVICE"] = device
    # create accelerator
    accelerator = Accelerator()
    print(accelerator.device)

    'read test data'
    # test = pd.read_json('datas/mag_sample.json', lines=True)
    # print(test.head())
    # print(test.shape)
    # test_human = test['abstract'].values.tolist()
    #
    # test = pd.read_json('datas/generated_abstracts_gpt3.json', lines=True)
    # print(test.head())
    # print(test.shape)
    # test_ai = test['generated_abstract'].values.tolist()
    #
    # test_texts = np.array(test_human + test_ai)
    # test_labels = np.array([0] * len(test_human) + [1] * len(test_ai))
    # test_ids = np.array(range(len(test_labels)))
    #
    # test_df = pd.DataFrame(data={'id': test_ids, 'text': test_texts})
    # print(test_df.head())
    # print(test_texts.shape, test_labels.shape, test_ids.shape)
    #
    # test_df = test_df[~test_df['text'].isnull()].copy()
    # test_df = test_df.reset_index(drop=True)
    # print(test_df.head())
    # print(test_df.shape)
    test_df = df.drop(columns=["generated"], axis=1)

    accelerator.print("~~" * 40)
    accelerator.print(f"PRE-PROCESSING: {cfg.preprocess_strategy.upper()}")
    accelerator.print("~~" * 40)

    test_df['text'] = test_df['text'].apply(lambda x: preprocess_text(x, cfg.preprocess_strategy))
    test_df = test_df.dropna().reset_index(drop=True)
    print(test_df)
    accelerator.print(f'Test csv shape: {test_df.shape}')

    with accelerator.main_process_first():
        dataset_creator = AiDataset(cfg)
        infer_ds = dataset_creator.get_dataset(test_df)

    tokenizer = dataset_creator.tokenizer
    # tokenizer.pad_token = tokenizer.eos_token

    infer_ds = infer_ds.sort("input_length")
    infer_ds.set_format(
        type=None,
        columns=[
            'id',
            'input_ids',
            'attention_mask',
        ]
    )
    print(infer_ds['input_ids'][:5])

    infer_ids = infer_ds["id"]  # .tolist()

    # --
    data_collator = AiCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=64
    )

    infer_dl = DataLoader(
        infer_ds,
        batch_size=cfg.predict_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    accelerator.print("data preparation done...")
    accelerator.print("~~" * 40)
    accelerator.wait_for_everyone()

    # ----------
    for b in infer_dl:
        break
    show_batch(b, tokenizer, task='infer', print_fn=accelerator.print)
    accelerator.print("~~" * 40)
    # ----------

    ## Load Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    if 'llama' in cfg.model.backbone_path.lower():
        base_model = LlamaForDetectAI.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,  # 2
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map=device
        )
    else:
        base_model = MistralForDetectAI.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map='cuda:1'
        )

    base_model.config.pretraining_tp = 1
    # base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, cfg.model.lora_path)
    accelerator.print("### Loaded Model Weights ###")

    model, infer_dl = accelerator.prepare(model, infer_dl)

    # run inference ---
    sub_df = run_inference(accelerator, model, infer_dl, infer_ids)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_path = os.path.join(save_dir, f"{model_id}.parquet")
        sub_df.to_parquet(save_path)
        accelerator.print("done!")
        accelerator.print("~~" * 40)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    ap.add_argument('--save_dir', type=str, required=True)
    ap.add_argument('--model_id', type=str, required=True)

    args = ap.parse_args()
    cfg = OmegaConf.load(args.config_path)

    os.makedirs(args.save_dir, exist_ok=True)

    # execution
    main(
        cfg,
        save_dir=args.save_dir,
        model_id=args.model_id,
    )