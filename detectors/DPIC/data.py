import pandas as pd
import numpy as np
import tqdm


def get_df(args):
    if args.mode == 'train':
        datasets_dict = {
            'cross-domain': ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
            'cross-model': ['llama', 'deepseek', 'gpt4o', 'Qwen'],
            'cross-operation': ['create', 'polish', 'translate']}
        datasets = datasets_dict[args.task]

        train_dir = f'./data/{args.task}'
        paths = [f'{train_dir}/{x}_sample.csv' for x in datasets if x != args.dataset]
        print(paths)

        datas = []
        for path in tqdm.tqdm(paths):
            original = pd.read_csv(path)
            text = original['text'].values.tolist()
            generated = original['generated'].values.tolist()
            datas.extend(zip(text, generated))

        df = pd.DataFrame(datas, columns=['text', 'generated'])
        df = get_sample(df,args.n_sample)

    else:
        if args.multilen>0:
            test_dir = f'./data/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
            df = pd.read_csv(test_dir)
            df['text'] = df[f'text_{args.multilen}']
        else:
            test_dir = f'./data/{args.task}/ori'
            path = f'{test_dir}/{args.dataset}_sample.csv'
            df = pd.read_csv(path)

    print(df.head())
    print(df.shape)

    return df


def get_dpic_df(args):
    if args.mode == 'train':
        datasets_dict = {
            'cross-domain': ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
            'cross-model': ['llama', 'deepseek', 'gpt4o', 'Qwen'],
            'cross-operation': ['create', 'polish', 'translate']}
        datasets = datasets_dict[args.task]

        train_dir = f'./detectors/DPIC/dpic_data/{args.task}/train'

        paths = [f'{train_dir}/{x}_sample.json' for x in datasets if x != args.dataset]
        print(paths)

        datas = []
        for path in tqdm.tqdm(paths):
            original = pd.read_json(path, lines=True)
            text = original['text'].values.tolist()
            generated_text = original['generated_text'].values.tolist()
            generated = original['generated'].values.tolist()
            datas.extend(zip(text, generated_text, generated))

        df = pd.DataFrame(datas, columns=['text', 'generated_text', 'generated'])
        df = get_sample(df,args.n_sample)

    elif args.mode == 'test':
        path = f'./detectors/DPIC/dpic_data/{args.task}/test/{args.dataset}_sample.json'
        print(path)
        df = pd.read_json(path, lines=True)
        if args.multilen > 0:
            df = run_datasplit(df, args.multilen)
    else:
        raise ValueError(f'Mode is not supported')

    print(df.head())
    print(df.shape)

    return df


def word_split(text):
    try:
        text_list = text.split()
    except Exception as e:
        text_list = None
    return text_list


def safe_join(x):
    if x is None:
        return ''
    else:
        return ' '.join(x)


def data_split(text, max_len):
    words = word_split(text)
    words_cutlen = words[:max_len] if words is not None else None
    text_cutlen = safe_join(words_cutlen)
    return text_cutlen


def run_datasplit(df, max_len):
    df['text'] = df['text'].apply(lambda x: data_split(x, max_len))
    df['generated_text'] = df['generated_text'].apply(lambda x: data_split(x, max_len))
    print(df.head())
    print(df.shape)
    return df


def get_sample(df,n_sample):
    df_human = df[df['generated'] == 0].reset_index(drop=True)
    df_ai = df[df['generated'] == 1].reset_index(drop=True)
    assert len(df_human) == len(df_ai)
    try:
        data_df = pd.DataFrame(
            {'original': df_human['text'], 'original_gen': df_human['generated_text'], 'rewritten': df_ai['text'],
             'rewritten_gen': df_ai['generated_text']}).dropna().reset_index(drop=True)
    except:
        data_df = pd.DataFrame(
            {'original': df_human['text'], 'rewritten': df_ai['text']}).dropna().reset_index(drop=True)
    data_df = data_df.sample(n=n_sample, random_state=12)

    sample_human = data_df['original'].values.tolist()
    sample_ai = data_df['rewritten'].values.tolist()
    sample_text = sample_human + sample_ai
    generated = [0] * len(sample_human) + [1] * len(sample_ai)
    text_id = list(range(len(sample_text)))
    try:
        sample_human_gen = data_df['original_gen'].values.tolist()
        sample_ai_gen = data_df['rewritten_gen'].values.tolist()
        sample_text_gen = sample_human_gen + sample_ai_gen
        sample_df = pd.DataFrame({'id': text_id, 'text': sample_text, 'generated_text': sample_text_gen, 'generated': generated})
    except:
        sample_df = pd.DataFrame({'id': text_id, 'text': sample_text, 'generated': generated})
    return sample_df