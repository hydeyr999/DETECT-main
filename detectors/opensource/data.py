import pandas as pd

def get_df(args):
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