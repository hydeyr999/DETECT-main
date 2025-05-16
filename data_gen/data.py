import pandas as pd
import tqdm
import ast

def convert_context_to_dict(context):
    start_idx = context.find("'contexts':")
    end_idx = context.find(", 'labels':")

    content = context[start_idx + len("'contexts':"):end_idx].strip()

    cleaned_text = content.replace('\n', '').replace('\r', '').strip()
    cleaned_text = cleaned_text.split('dtype=object')[0]
    cleaned_text = cleaned_text[len("array(["):].strip()
    cleaned_text = cleaned_text.rstrip("],")

    content_list = ast.literal_eval(cleaned_text)

    result = ''
    for str in content_list:
        result = result + str

    return result

def get_df(args):
    if args.dataset == 'pubmedqa':
        df = pd.read_csv('./data_gen/human/pubmedqa_human_3.5k.csv')
        context = df['context'].values
        human_text = [convert_context_to_dict(x) for x in tqdm.tqdm(context)]
        df['human_text'] = human_text
        df = df[['human_text']]
        print(df.head())

    elif args.dataset == 'writingprompts':
        df = pd.read_csv('./data_gen/human/wp_human_3.5k.csv')
        df = df[['story']]
        print(df.head())

    elif args.dataset == 'squad':
        df = pd.read_csv('./data_gen/human/squad_human_3.5k.csv')
        df = df[['context']]
        print(df.head())

    elif args.dataset == 'xsum':
        df = pd.read_csv('./data_gen/human/xsum_human_3.5k.csv')
        df = df[['document']]
        print(df.head())

    elif args.dataset == 'openreview':
        df = pd.read_csv('./data_gen/human/openreview_human_3.5k.csv')
        df = df[['text']]
        print(df.head())

    elif args.dataset == 'tweets':
        df = pd.read_csv('./data_gen/human/tweets_human_3.5k.csv')
        df = df[['text']]
        print(df.head())

    elif args.dataset == 'blog':
        df = pd.read_csv('./data_gen/human/blog_human_3.5k.csv')
        df = df[['text']]
        print(df.head())

    else:
        raise ValueError(f"Invalid dataset name.")

    return df