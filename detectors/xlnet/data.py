import pandas as pd
import tqdm
import os

def get_df(args):
    if args.mode =='train':
        datasets_dict = {
            'cross-domain': ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
            'cross-detectors': ['llama', 'deepseek', 'gpt4o','Qwen'],
            'cross-operation': ['create', 'translate', 'polish', 'expand','refine','summary','rewrite']}
        datasets = datasets_dict[args.task]
        paths = [f'./data/{args.task}/{x}_sample.csv' for x in datasets if x != args.dataset]
        print(paths)

        datas = []
        for path in tqdm.tqdm(paths):
            original = pd.read_csv(path)
            text = original['text'].values.tolist()
            generated = original['generated'].values.tolist()
            datas.extend(zip(text, generated))

        df = pd.DataFrame(datas, columns=['text', 'generated'])
        df['id'] = range(len(df))

        df = get_sample(df,args.n_sample)

    else:
        if args.multilen>0:
            test_dir = f'./data/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
            df = pd.read_csv(test_dir)
            df['text'] = df[f'text_{args.multilen}']
        else:
            test_dir = f'./data/{args.task}'
            path = f'{test_dir}/{args.dataset}_sample.csv'
            df = pd.read_csv(path)

    print(df)
    print(df.shape)

    return df

def get_sample(df,n_sample=2000):
    df_human = df[df['generated'] == 0].reset_index(drop=True)
    df_ai = df[df['generated'] == 1].reset_index(drop=True)
    assert len(df_human) == len(df_ai)
    data_df = pd.DataFrame({'original': df_human['text'], 'rewritten': df_ai['text']}).dropna().reset_index(drop=True)
    data_df = data_df.sample(n=n_sample, random_state=12)

    sample_human = data_df['original'].values.tolist()
    sample_ai = data_df['rewritten'].values.tolist()
    sample_text = sample_human + sample_ai
    generated = [0] * len(sample_human) + [1] * len(sample_ai)
    id = list(range(len(sample_text)))
    sample_df = pd.DataFrame({'id': id, 'text': sample_text, 'generated': generated})

    return sample_df