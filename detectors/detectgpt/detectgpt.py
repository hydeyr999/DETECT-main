import argparse
import json

import numpy as np
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import functools
import time
import gc
from data import *
from metrics import *


pattern = re.compile(r"<extra_id_\d+>")

def strip_newlines(text):
    return ' '.join(text.split())

def get_ll(args,text,base_model,base_tokenizer):
    with torch.no_grad():
        encodings = base_tokenizer(text, return_tensors="pt",truncation=True,padding = 'max_length',max_length=512).to(args.device)
        logits = F.softmax(base_model(encodings["input_ids"]).logits, dim=2)

        tokens = encodings["input_ids"]
        indices = torch.tensor([[[i] for i in tokens[0]]])[:, 1:, :].to(args.device)

        subprobs = (
            torch.gather(logits[:, :-1, :], dim=2, index=indices)
            .flatten()
            .cpu()
            .detach()
            .numpy()
        )
    del encodings,logits, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return np.mean(subprobs)

def get_lls(args,texts,base_model,base_tokenizer):
    probs = []
    for text in texts:
        logprob = get_ll(args,text,base_model,base_tokenizer)
        probs.append(logprob)
    return probs

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

def replace_masks(args,texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True,max_length=512,truncation=True).to(args.device)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    texts = [" ".join(x) for x in tokens]
    return texts

def tokenize_and_mask(text, span_length, pct, buffer_size,ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def perturb_texts_(args,texts, span_length, pct,buffer_size, ceil_pct=False):
    if not args.random_fills:
        masked_texts = [tokenize_and_mask(x, span_length, pct, buffer_size, ceil_pct) for x in texts]
        raw_fills = replace_masks(args,masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(args,masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
            if attempts > 3:
                break
    else:
        return

    return perturbed_texts

def perturb_texts(texts,args, span_length, pct,buffer_size, ceil_pct=False):
    chunk_size = 10
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(args,texts[i:i + chunk_size], span_length, pct,buffer_size, ceil_pct=ceil_pct))
    return outputs

def get_perturbation_results(args, base_model, base_tokenizer,span_length=10, n_perturbations=1):
    load_mask_model(args)

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    if args.no_generate:
        original_text = data
        perturb_fn = functools.partial(perturb_texts, args=args,span_length=span_length, pct=args.pct_words_masked,buffer_size=args.buffer_size)

        p_original_text = perturb_fn([x for x in original_text for _ in range(n_perturbations)])
        for _ in range(args.n_perturbation_rounds - 1):
            try:
                p_original_text = perturb_fn(p_original_text)
            except AssertionError:
                break

        assert len(p_original_text) == len(
            original_text) * n_perturbations, f"Expected {len(original_text) * n_perturbations} perturbed samples, got {len(p_original_text)}"

        for idx in range(len(original_text)):
            results.append({
                "original": original_text[idx],
                "perturbed_original": p_original_text[idx * n_perturbations: (idx + 1) * n_perturbations]
            })

        load_base_model(args)

        for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
            p_original_ll = get_lls(args,res["perturbed_original"],base_model,base_tokenizer)
            res["original_ll"] = get_ll(args,res["original"],base_model,base_tokenizer)
            res["all_perturbed_original_ll"] = p_original_ll
            res["perturbed_original_ll"] = np.mean(p_original_ll)
            res["perturbed_original_ll_std"] = np.std(p_original_ll) if len(p_original_ll) > 1 else 1
    else:
        return

    return results

def run_perturbation_experiment(results, criterion,data_label=None, span_length=10, n_perturbations=1, n_samples=500):
    # compute diffs with perturbed
    if args.no_generate:
        predictions = []
        for res in results:
            if criterion == 'd':
                predictions.append(res['original_ll'] - res['perturbed_original_ll'])
            elif criterion == 'z':
                if res['perturbed_original_ll_std'] == 0:
                    res['perturbed_original_ll_std'] = 1
                predictions.append(
                    (res['original_ll'] - res['perturbed_original_ll']) / res['perturbed_original_ll_std'])

        fpr, tpr, roc_auc = get_roc_metrics(predictions,data_label)
        p, r, pr_auc = get_precision_recall_metrics(predictions, data_label)
    else:
        return

    name = f'perturbation_{n_perturbations}_{criterion}'
    print(f"{name} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
    return {
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


def generate_data(args,dataset, key):
    # load data
    data = dataset[key]
    data_label = dataset['generated']
    print(len(data),len(data_label))

    data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

    # strip whitespace around each example
    data = [str(x).strip() for x in data]

    # remove newlines from each example
    data = [strip_newlines(x) for x in data]

    data = np.array(data)
    data_label = np.array(data_label)

    indices = np.array(range(len(data)))
    random.seed(0)
    random.shuffle(indices)
    print(f"Shuffling data......")
    print('indices:', indices)
    data = data[indices].tolist()
    data_label = data_label[indices].tolist()


    print(f"Total number of samples: {len(data)}")
    print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

    return data[:args.n_samples],data_label[:args.n_samples]

def trim_to_shorter_length(texta, textb):
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb

def sample_from_model(texts, min_words=55, prompt_tokens=30):
    all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(args.device)
    all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

    decoded = ['' for _ in range(len(texts))]

    tries = 0
    while (m := min(len(x.split()) for x in decoded)) < min_words:
        if tries != 0:
            print()
            print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

        sampling_kwargs = {}
        if args.do_top_p:
            sampling_kwargs['top_p'] = 0.96
        elif args.do_top_k:
            sampling_kwargs['top_k'] = 40
        min_length = 150
        outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
        decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        tries += 1

    return decoded

def load_base_model(args):
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    base_model.to(args.device)
    print(f'DONE ({time.time() - start:.2f}s)')

def load_mask_model(args):
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    base_model.cpu()
    if not args.random_fills:
        mask_model.to(args.device)
    print(f'DONE ({time.time() - start:.2f}s)')


def clean_text(data):
    data = list(dict.fromkeys(data))

    data = [x.strip() for x in data]

    data = [strip_newlines(x) for x in data]
    tokenized_data = preproc_tokenizer(data)
    data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]
    return data

def custom_serializer(obj):
    if isinstance(obj, np.float16):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_filling_model_name', type=str, default='t5-base', help='Name of the mask filling model')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--n_perturbation_list', type=int, nargs='+', default=[10],help='List of perturbation counts')
    parser.add_argument('--n_perturbation_rounds', type=int, default=1, help='Number of perturbation rounds')
    parser.add_argument('--n_similarity_samples', type=int, default=20, help='Number of similarity samples')
    parser.add_argument('--pct_words_masked', type=float, default=0.3, help='Percentage of words to mask')
    parser.add_argument('--do_top_p', type=bool, default=False, help='Whether to apply top_p')
    parser.add_argument('--do_top_k', type=bool, default=False, help='Whether to apply top_k')
    parser.add_argument('--random_fills', type=bool, default=False, help='Whether to use random fills')
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0, help='Pre perturbation percentage')
    parser.add_argument('--pre_perturb_span_length', type=int, default=5, help='Pre perturbation span length')
    parser.add_argument('--chunk_size', type=int, default=20, help='Chunk size')
    parser.add_argument('--buffer_size', type=int, default=1, help='Buffer size')
    parser.add_argument('--mask_top_p', type=float, default=1.0, help='Mask top p')
    parser.add_argument('--no_generate', type=bool, default=True, help='Whether to disable generation')
    parser.add_argument('--base_model', type=str, default='./models/gpt2-medium', help="Base model path.")
    parser.add_argument('--mask_model', type=str, default='./models/t5-base', help="Mask model path.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device")
    parser.add_argument("--task",type=str,choices= ['cross-model','cross-domain','cross-operation'])
    parser.add_argument("--dataset",type=str,default=None)
    parser.add_argument("--multilen",type=int,default=0)
    args = parser.parse_args()

    device_map = {'' : int(args.device.split(':')[1])}
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )
    base_model = transformers.AutoModelForCausalLM.from_pretrained(f'{args.base_model}/model',
                                                                   quantization_config=bnb_config,
                                                                   use_cache=False,
                                                                   device_map=device_map)
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(f'{args.base_model}/tokenizer')
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    print(f'Loading mask filling model {args.mask_filling_model_name}...')
    mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.mask_model)
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512

    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_model, model_max_length=512)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(args.mask_model, model_max_length=n_positions)

    load_base_model(args)

    print(f'Loading dataset {args.dataset}_essays.csv...')
    df = get_df(args).dropna().reset_index(drop=True).sample(n=5)
    print(df)

    data, data_label = generate_data(args,df, 'text')

    outputs = []
    for n_perturbations in args.n_perturbation_list:
        perturbation_results = get_perturbation_results(args,base_model,base_tokenizer,2, n_perturbations)
        for perturbation_mode in ['d', 'z']:
            output = run_perturbation_experiment(
                perturbation_results, perturbation_mode, data_label=data_label, span_length=2,
                n_perturbations=n_perturbations,
                n_samples=args.n_samples)
            outputs.append(output)

    detect_save_dir = f'./detectors/detectgpt/results/{args.task}'
    if not os.path.exists(detect_save_dir):
        os.mkdir(detect_save_dir)
    with open(f'{detect_save_dir}/detectgpt_{args.dataset}_multi{args.multilen}.json', 'w') as f:
        json.dump(outputs, f, default=custom_serializer)
