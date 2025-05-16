# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
import tqdm
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import json
from metrics import get_roc_metrics, get_precision_recall_metrics

from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AdamW
import time

def evaluate_model_SPO(model, data, DEVICE):
    model.to(DEVICE)
    model.eval()
    loss = 0
    eval_loader = DataLoader(data, batch_size=1, shuffle=False)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm.tqdm(eval_loader, desc="Evaluating"):
            text = batch
            output = model(text)
            loss += output['loss'].item()
            epoch_crit_train_original.extend(output['crit'][1].tolist())
            epoch_crit_train_sampled.extend(output['crit'][3].tolist())
            
        print(f"Total time: {time.time() - start_time:.4f}s")
        avg_loss = loss / len(eval_loader)
        fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
    
    print(f"val_loss: {avg_loss:.6f}")
    print(f"val_ROC_AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    print(f"val_Real_mean/std: {np.mean(epoch_crit_train_original):.2f}/{np.std(epoch_crit_train_original):.2f}, val_Samples_mean/std: {np.mean(epoch_crit_train_sampled):.2f}/{np.std(epoch_crit_train_sampled):.2f}")
    print("="*10)
    
    results_dict = {
        "name": "SPO",
        'val_loss': avg_loss,
        'val_ROC_AUC': roc_auc,
        'val_PR_AUC': pr_auc,
        'val_crit_real_mean': np.mean(epoch_crit_train_original),
        'val_crit_real_std': np.std(epoch_crit_train_original),
        'val_crit_sampled_mean': np.mean(epoch_crit_train_sampled),
        'val_crit_sampled_std': np.std(epoch_crit_train_sampled),
        'crit_real': epoch_crit_train_original,
        'crit_sampled': epoch_crit_train_sampled,
        'fpr': fpr,
        'tpr': tpr,
    }
    return results_dict


def fine_tune_ours(model, data, DEVICE, ckpt_dir='./ckpt', args=None):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"./detectors/imbd/log/{args.task_name}/{args.train_dataset}_spo_lr_{args.lr}_beta_{args.beta}_a_{args.a}_{current_time}/train_ai_detection")

    train_loader = DataLoader(data[0], batch_size=1, shuffle=True)
    epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=0,
                                  last_epoch=-1)

    scaler = GradScaler()
    model.to(DEVICE)

    # Number of iterations for gradient accumulation
    accumulation_steps = args.a
    epoch_losses, i, loss = [], 0, torch.tensor(0.0).to(DEVICE)
    epoch_crit_train_original, epoch_crit_train_sampled = [],[]
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        start_time = time.time()
        for batch in tqdm.tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            try:
                text = batch
                scheduler.step()
                with autocast():
                    outputs_1 = model(text)
                    epoch_crit_train_original.extend(outputs_1['crit'][1].tolist())
                    epoch_crit_train_sampled.extend(outputs_1['crit'][3].tolist())
                    loss += (outputs_1['loss'].to(torch.float32)) / accumulation_steps


                if ((i + 1) % accumulation_steps) == 0:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                    writer.add_scalar('Loss/train', loss.item(), i)
                    epoch_losses.append(loss.item())
                    loss = torch.tensor(0.0).to(DEVICE)
                epoch_losses.append(loss.item())
                i += 1
            except torch.OutOfMemoryError as e:  # 捕获显存不足异常
                torch.cuda.empty_cache()  # 清理显存
                continue
            except Exception as e:
                print(f"Exception occurred: {e}")
        print(f"Total time: {time.time() - start_time:.4f}s") 
        fpr, tpr, roc_auc = get_roc_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        p, r, pr_auc = get_precision_recall_metrics(epoch_crit_train_original, epoch_crit_train_sampled)
        
        print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        print(f"Real mean/std: {np.mean(epoch_crit_train_original):.2f}/{np.std(epoch_crit_train_original):.2f}, Samples mean/std: {np.mean(epoch_crit_train_sampled):.2f}/{np.std(epoch_crit_train_sampled):.2f}")
        epoch_avg_loss = np.mean(epoch_losses)
        
        writer.add_scalar('Loss/epoch', epoch_avg_loss, epoch)
        writer.add_scalar('ROC_AUC/epoch', roc_auc, epoch)
        writer.add_scalar('PR_AUC/epoch', pr_auc, epoch)
        writer.add_scalar('Real_mean/epoch',np.mean(epoch_crit_train_original),epoch)
        writer.add_scalar('Real_std/epoch',np.std(epoch_crit_train_original),epoch)
        writer.add_scalar('Sampled_mean/epoch',np.mean(epoch_crit_train_sampled),epoch)
        writer.add_scalar('Sampled_std/epoch',np.std(epoch_crit_train_sampled),epoch)
        epoch_crit_train_original, epoch_crit_train_sampled = [],[] # reset crit
        print(f"\nAverage Loss for Epoch {epoch}: {epoch_avg_loss}")
        
        if ((epoch+1) % args.val_freq)==0: # eval and save model every val_freq epoch
            
            dataset_name = args.eval_dataset.split("/")[-1].split("_")[0]
            print(f"Evaluating on {dataset_name}...")
            val_result = evaluate_model_SPO(model, data[1], DEVICE)
            valwriter = SummaryWriter(log_dir=f"./detectors/imbd/logs/{args.task_name}/{args.eval_dataset}_spo_lr_{args.lr}_beta_{args.beta}_a_{args.a}_{current_time}/val_{dataset_name}")
            valwriter.add_scalar('val_Loss/epoch', val_result['val_loss'], epoch)
            valwriter.add_scalar('val_ROC_AUC/epoch', val_result['val_ROC_AUC'], epoch)
            valwriter.add_scalar('val_PR_AUC/epoch', val_result['val_PR_AUC'], epoch)
            valwriter.add_scalar('val_Real_mean/epoch', val_result['val_crit_real_mean'], epoch)
            valwriter.add_scalar('val_Real_std/epoch', val_result['val_crit_real_std'], epoch)
            valwriter.add_scalar('val_Sampled_mean/epoch', val_result['val_crit_sampled_mean'], epoch)
            valwriter.add_scalar('val_Sampled_std/epoch', val_result['val_crit_sampled_std'], epoch)
            valwriter.close()

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.save_pretrained(ckpt_dir)
    print(f"Saved finetuned model to {os.path.join(ckpt_dir, 'ours-finetuned.pth')}")
    
    writer.close()
    return model


def run(
    model, 
    data, 
    DEVICE,
    args,
    ckpt_dir='./ckpt',
    ):
    print(DEVICE)
    if args.ebt or args.eval_only:
        print("Evaluating model before tuning...")
        dataset_name = args.eval_dataset
        d = evaluate_model_SPO(model, data[1], DEVICE)
        if args.SPOtrained:
            output_path = os.path.join(args.output_file, f"{args.task_name}/imbd_{dataset_name}_len{args.cut_len}_n{args.datanum}.json")
        else:
            method_name = args.base_model.split("_")[-1]
            output_path = os.path.join(args.output_file, f"{args.task_name}/imbd_{dataset_name}_len{args.cut_len}_n{args.datanum}.json")
        with open(output_path, "w") as j:
            json.dump(d, j)
        print(f"Results saved to {output_path}.")
    if args.eval_only:
        return

    print('Fine-tuning model...')
    model = fine_tune_ours(
        model, 
        data,
        DEVICE=DEVICE, 
        ckpt_dir=ckpt_dir,
        args=args
    )
    



