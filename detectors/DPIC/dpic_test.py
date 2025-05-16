import generate
from model import *
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch
import os
import tqdm
from data import *
from sklearn.metrics import roc_auc_score, roc_curve

def get_eval(probs_df, df):
    roc_auc = roc_auc_score(df.generated.values, probs_df.generated.values)
    print('roc_auc:', roc_auc)
    fpr, tpr, thresholds = roc_curve(df.generated.values, probs_df.generated.values)
    fpr_tar = fpr[np.where(tpr >= 0.95)][0]
    tpr_tar = tpr[np.where(fpr <= 0.1)][-1]
    print('fpr:', fpr_tar)
    print('tpr:', tpr_tar)

def get_prediction(model,tokenizer, df,device='cuda',max_len=768,batch_size=16):
    model.to(device)
    model.eval()

    test_dataset = AIDataset(df, tokenizer, max_len)
    test_generator = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False)

    pred_prob = np.zeros((len(df),), dtype=np.float32)

    for j, (input_ids_ori, input_ids_gen, attention_mask, _) in tqdm.tqdm(enumerate(test_generator),total=len(test_generator)):
        with torch.no_grad():
            start = j * batch_size
            end = start + batch_size
            if j == len(test_generator) - 1:
                end = len(test_generator.dataset)

            input_ids_ori = input_ids_ori.to(device)
            input_ids_gen = input_ids_gen.to(device)
            attention_mask = attention_mask.to(device)

            with autocast():
                logits = model(input_ids_ori, input_ids_gen, attention_mask)
            pred_prob[start:end] = logits.sigmoid().cpu().data.numpy().squeeze()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return pred_prob

def get_test_result(args,model,tokenizer, df, output_dir):
    bert_prob0 = get_prediction(model,
                                tokenizer,
                                df,
                                max_len=args.max_len,
                                device=args.device)

    print(bert_prob0)
    gc.collect()
    torch.cuda.empty_cache()

    bert_probs_df = pd.DataFrame(data={'id': df.id.values, 'generated': bert_prob0})
    bert_probs_df.to_csv(output_dir, index=False)
    bert_probs_df = pd.read_csv(output_dir)
    print('bert_probs_df:', bert_probs_df)

    get_eval(bert_probs_df, df)


def run_dpic_test(args):
    if args.run_generate:
        test_df = generate.run(args)
        if args.gen_save_path:
            test_df.to_json(args.gen_save_path, orient='records', lines=True)
        else:
            test_df.to_json(f'{args.data_save_dir}/{args.task}/test/{args.dataset}_sample.json', orient='records', lines=True)
        return
    else:
        test_df = get_dpic_df(args).dropna()

    model_path = args.backbone
    model_weights = args.dpic_ckpt
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = DPICModel(model_path, config, tokenizer, pretrained=False)
    model.load_state_dict(torch.load(model_weights))

    get_test_result(args,model,tokenizer, test_df,
                    f'./detectors/DPIC/results/{args.task}/dpic_{args.dataset}_ep{args.epochs}_multi{args.multilen}_n{args.n_sample}.csv')