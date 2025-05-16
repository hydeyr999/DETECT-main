import os

import numpy as np
import pandas as pd
import torch
import tqdm
from collections import defaultdict, Counter
from transformers import PreTrainedTokenizerBase,AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nltk.corpus import brown
from nltk import ngrams
import gc
import argparse
from functions import *
import joblib
import json
from utils.featurize import normalize
from utils.symbolic import vec_functions, scalar_functions

vectors = ["llama-logprobs", "unigram-logprobs", "trigram-logprobs"]
best_features = [
    "trigram-logprobs v-add unigram-logprobs v-> llama-logprobs s-var",
    "trigram-logprobs v-div unigram-logprobs v-div trigram-logprobs s-avg-top-25",
    "unigram-logprobs v-mul llama-logprobs s-avg",
    "trigram-logprobs v-mul unigram-logprobs v-div trigram-logprobs s-avg",
    "trigram-logprobs v-< unigram-logprobs v-mul llama-logprobs s-avg-top-25",
    "trigram-logprobs v-mul unigram-logprobs v-sub llama-logprobs s-min",
    "trigram-logprobs v-mul unigram-logprobs s-avg",
    "trigram-logprobs v-< unigram-logprobs v-sub llama-logprobs s-avg",
    "trigram-logprobs v-> unigram-logprobs v-add llama-logprobs s-avg",
    "trigram-logprobs v-div llama-logprobs v-div trigram-logprobs s-min",
]

def get_words(exp):
    return exp.split(" ")

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

"""
run ghostbuster
"""
def calc_features(idx, exp, vector_map):
    exp_tokens = get_words(exp)
    curr = vector_map[exp_tokens[0]](idx)

    for i in range(1, len(exp_tokens)):
        if exp_tokens[i] in vec_functions:
            next_vec = vector_map[exp_tokens[i + 1]](idx)
            curr = vec_functions[exp_tokens[i]](curr, next_vec)
        elif exp_tokens[i] in scalar_functions:
            return scalar_functions[exp_tokens[i]](curr)

def exp_featurize(idx,vector_map):
    return np.array([calc_features(idx, exp, vector_map) for exp in best_features])

class NGramModel:
    """
    An n-gram model, where alpha is the laplace smoothing parameter.
    """

    def __init__(self, train_text, n=2, alpha=3e-3, vocab_size=None):
        self.n = n
        if vocab_size is None:
            # Assume GPT tokenizer
            self.vocab_size = 50257
        else:
            self.vocab_size = vocab_size

        self.smoothing = alpha
        self.smoothing_f = alpha * self.vocab_size

        self.c = defaultdict(lambda: [0, Counter()])
        for i in tqdm.tqdm(range(len(train_text) - n)):
            n_gram = tuple(train_text[i : i + n])
            self.c[n_gram[:-1]][1][n_gram[-1]] += 1
            self.c[n_gram[:-1]][0] += 1
        self.n_size = len(self.c)

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == self.n
        it = self.c[tuple(n_gram[:-1])]
        prob = (it[1][n_gram[-1]] + self.smoothing) / (it[0] + self.smoothing_f)
        return prob
class DiscountBackoffModel(NGramModel):
    """
    An n-gram model with discounting and backoff. Delta is the discounting parameter.
    """

    def __init__(self, train_text, lower_order_model, n=2, delta=0.9, vocab_size=None):
        super().__init__(train_text, n=n, vocab_size=vocab_size)
        self.lower_order_model = lower_order_model
        self.discount = delta

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == self.n
        it = self.c[tuple(n_gram[:-1])]

        if it[0] == 0:
            return self.lower_order_model.n_gram_probability(n_gram[1:])

        prob = (
            self.discount
            * (len(it[1]) / it[0])
            * self.lower_order_model.n_gram_probability(n_gram[1:])
        )
        if it[1][n_gram[-1]] != 0:
            prob += max(it[1][n_gram[-1]] - self.discount, 0) / it[0]

        return prob
class KneserNeyBaseModel(NGramModel):
    """
    A Kneser-Ney base model, where n=1.
    """

    def __init__(self, train_text, vocab_size=None):
        super().__init__(train_text, n=1, vocab_size=vocab_size)

        base_cnt = defaultdict(set)
        for i in range(1, len(train_text)):
            base_cnt[train_text[i]].add(train_text[i - 1])

        cnt = 0
        for word in base_cnt:
            cnt += len(base_cnt[word])

        self.prob = defaultdict(float)
        for word in base_cnt:
            self.prob[word] = len(base_cnt[word]) / cnt

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == 1
        ret_prob = self.prob[n_gram[0]]

        if ret_prob == 0:
            return 1 / self.vocab_size
        else:
            return ret_prob
class TrigramBackoff:
    """
    A trigram model with discounting and backoff. Uses a Kneser-Ney base model.
    """

    def __init__(self, train_text, delta=0.9, vocab_size=None):
        self.base = KneserNeyBaseModel(train_text, vocab_size=vocab_size)
        self.bigram = DiscountBackoffModel(
            train_text, self.base, n=2, delta=delta, vocab_size=vocab_size
        )
        self.trigram = DiscountBackoffModel(
            train_text, self.bigram, n=3, delta=delta, vocab_size=vocab_size
        )

    def n_gram_probability(self, n_gram):
        assert len(n_gram) == 3
        return self.trigram.n_gram_probability(n_gram)

def score_ngram(doc, model, tokenizer, n=3, strip_first=False, bos_token_id=50256):
    """
    Returns vector of ngram probabilities given document, model and tokenizer

    Slightly modified from here: https://github.com/vivek3141/ghostbuster/blob/9831b53a8ecbfe401d47616db95b9256b9cbaadd/utils/featurize.py#L65-L75
    """
    scores = []
    if strip_first:
        doc = " ".join(doc.split()[:1000])

    if isinstance(tokenizer.__self__, PreTrainedTokenizerBase):
        tokens = tokenizer(doc.strip().replace('\n',''), add_special_tokens=True,truncation=True,padding = 'max_length',max_length=512)
        # tokens[0] is bos token
        tokens = (n - 1) * [tokens[0]] + tokens
    else:
        eos_token_id = 50256  # eos/bos token for davinci model
        tokens = (n - 1) * [eos_token_id] + tokenizer(doc.strip())
    # print(len(tokens))
    # for k tokens and ngrams of size n, need to add n-1 tokens to the beginning
    # to ensure that there are k ngrams
    for i in ngrams(tokens, n):
        scores.append(model.n_gram_probability(i))

    return np.array(scores)
def train_trigram(enc, verbose=True, return_tokenizer=False):
    tokenizer = enc.encode
    sentences = brown.sents()
    if verbose:
        print("Tokenizing corpus...")
    tokenized_corpus = []
    for sentence in tqdm.tqdm(sentences):
        tokens = tokenizer(" ".join(sentence))
        tokenized_corpus += tokens

    if verbose:
        print("\nTraining n-gram model...")

    if return_tokenizer:
        return TrigramBackoff(tokenized_corpus), tokenizer
    else:
        return TrigramBackoff(tokenized_corpus)

def get_ghost_features(df, tokenizer,max_length):
    test_texts = df['text']
    llama_logprobs = df['logprob_large']
    trigram_model = train_trigram(tokenizer)
    unigram_logprobs = []
    trigram_logprobs = []
    for idx in tqdm.tqdm(range(len(test_texts))):
        trigram = np.array(score_ngram(test_texts[idx], trigram_model, tokenizer.encode, n=3, strip_first=False))[1:max_length]
        unigram = np.array(score_ngram(test_texts[idx], trigram_model.base, tokenizer.encode, n=1, strip_first=False))[1:max_length]
        unigram_logprobs.append(unigram)
        trigram_logprobs.append(trigram)

    vector_map = {
        'llama-logprobs': lambda x: llama_logprobs[x],
        'trigram-logprobs': lambda x: trigram_logprobs[x],
        'unigram-logprobs': lambda x: unigram_logprobs[x],
    }
    datas = np.vstack([exp_featurize(i,vector_map) for i in range(len(df))])
    datas = normalize(datas)

    gc.collect()
    torch.cuda.empty_cache()
    return datas

def get_logprob(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_likelihood = log_likelihood.cpu().detach().numpy()

    return log_likelihood

def get_logprobs(model,tokenizer,test_df,max_length):
    texts = test_df['text'].values
    logprobs = []
    for text in tqdm.tqdm(texts):
        tokenized = tokenizer(text,
                              max_length=max_length,
                              truncation=True,
                              return_tensors="pt",
                              padding='max_length',
                              return_token_type_ids=False).to(model.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits = model(**tokenized).logits[:, :-1]
            logprob = get_logprob(logits, labels)
            logprobs.append(logprob)
    return logprobs

def run_ghostbuster(args,test_df= None):
    name_small = args.model_path_small.split('/')[-1]
    name_large = args.model_path_large.split('/')[-1]
    print(name_small, name_large)

    if args.mode == 'train':
        train_df = get_df(args,name_small,name_large).dropna().reset_index(drop=True)

        if args.run_logprobs:
            model_small, tokenizer = get_model(args.model_path_small,args.device)
            logprobs_small = get_logprobs(model_small,tokenizer,train_df,args.max_length)

            model_large, _ = get_model(args.model_path_large,args.device)
            logprobs_large = get_logprobs(model_large,tokenizer,train_df,args.max_length)
        else:
            logprobs_small = train_df['logprob_small']
            logprobs_large = train_df['logprob_large']

            tokenizer = AutoTokenizer.from_pretrained(args.model_path_small)
            tokenizer.pad_token = tokenizer.eos_token

        if args.if_save:
            logprobs_df = pd.DataFrame({'id': train_df['id'],
                                        'text': train_df['text'],
                                        'logprob_small': logprobs_small,
                                        'logprob_large': logprobs_large,
                                        'generated': train_df['generated']})
            logprobs_save_dir = f'./detectors/ling-based/logprobs'
            if args.task == 'otherdata':
                logprobs_df.to_json(f'{logprobs_save_dir}/{args.task}/{args.dataset}_{name_small}_{name_large}.json',orient="records", lines=True)
            else:
                logprobs_df.to_json(f'{logprobs_save_dir}/{args.task}/train/{args.dataset}_{name_small}_{name_large}.json',orient="records", lines=True)

        features = get_ghost_features(train_df, tokenizer,args.max_length)
        classifier = get_classifier(features, train_df['generated'])

        model_save_dir = f'./detectors/ling-based/classifier'
        joblib.dump(classifier, f'{model_save_dir}/{args.task}/gb_{args.dataset}_n{args.n_sample}.pkl')

    elif args.mode == 'test':
        test_df = get_df(args,name_small,name_large).dropna().reset_index(drop=True)
        print(test_df)

        if args.run_logprobs:
            model_small, tokenizer = get_model(args.model_path_small,args.device)
            logprobs_small = get_logprobs(model_small,tokenizer,test_df,args.max_length)

            model_large, _ = get_model(args.model_path_large,args.device)
            logprobs_large = get_logprobs(model_large,tokenizer,test_df,args.max_length)
        else:
            logprobs_small = test_df['logprob_small']
            logprobs_large = test_df['logprob_large']

            tokenizer = AutoTokenizer.from_pretrained(args.model_path_small)
            tokenizer.pad_token = tokenizer.eos_token

        if args.if_save:
            logprobs_df = pd.DataFrame({'id': test_df['id'],
                                        'text': test_df['text'],
                                        'logprob_small': logprobs_small,
                                        'logprob_large': logprobs_large,
                                        'generated': test_df['generated']})

            if test_df.columns.str.contains('model').any():
                logprobs_df['model'] = test_df['model']
            if test_df.columns.str.contains('operation').any():
                logprobs_df['operation'] = test_df['operation']
            if test_df.columns.str.contains('domain').any():
                logprobs_df['domain'] = test_df['domain']
            logprobs_save_dir = f'./detectors/ling-based/logprobs'
            logprobs_df.to_json(f'{logprobs_save_dir}/{args.task}/test/{args.dataset}_{name_small}_{name_large}.json',orient="records", lines=True)
            return

        features = get_ghost_features(test_df, tokenizer,args.max_length)
        classifier = joblib.load(args.classifier)

        predictions = classifier.predict_proba(features)[:,1]
        fpr, tpr, roc_auc = get_roc_metrics(test_df['generated'].values, predictions)
        p, r, pr_auc = get_precision_recall_metrics(test_df['generated'].values, predictions)
        print(f'ROC AUC: {roc_auc:.4f}')

        results_file = f'results/{args.task}/{args.dataset}_gb_{name_small}_{name_large}.json'
        results = {'predictions': predictions.tolist(),
                   'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                   'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                   'loss': 1 - pr_auc}

        res_save_dir = f'./detectors/ling-based/results/{args.task}'
        if not os.path.exists(res_save_dir):
            os.makedirs(res_save_dir)
        with open(f'{res_save_dir}/gb_{args.dataset}_n{args.n_sample}_multi{args.multilen}.json', 'w') as fout:
            json.dump(results, fout)
            print(f'Results written into {res_save_dir}/gb_{args.dataset}_n{args.n_sample}_multi{args.multilen}.json')
    else:
        raise ValueError('Invalid mode.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,required=True)
    parser.add_argument('--task',type=str,choices = ['cross-domain','cross-model','cross-operation','otherdata'])
    parser.add_argument('--model_path_small', type=str, default='./models/tinyllama')
    parser.add_argument('--model_path_large', type=str, default='./models/llama2_7b')
    parser.add_argument('--mode', type=str,default='test',choices=['test','train'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--run_logprobs', type=bool, default=False)
    parser.add_argument('--if_save', type=bool, default=False)
    parser.add_argument('--classifier', type=str, default=None)
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--multilen', type=int, default=0)
    args = parser.parse_args()

    run_ghostbuster(args)