import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/llama3-8b-v2')
    parser.add_argument('--backbone', type=str, default='./models/deberta-v3-base')
    parser.add_argument('--dpic_ckpt', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--task',type=str,choices = ['cross-domain','cross-operation','cross-model'],default='cross-domain')
    parser.add_argument('--dataset',type=str,default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--multilen',type=int,default=0)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--model_save_dir', type=str, default='./detectors/DPIC/weights')
    parser.add_argument('--data_save_dir', type=str, default='./detectors/DPIC/dpic_data')
    parser.add_argument('--run_generate', type=bool, default=False)
    parser.add_argument('--ori_data_path',type=str, default=None)
    parser.add_argument('--gen_save_path', type=str, default=None)
    parser.add_argument('--n_sample',type=int, default=0)

    args = parser.parse_args()

    if args.mode == 'train':
        from dpic_train import *
        run_dpic_train(args)

    elif args.mode == 'test':
        from dpic_test import *
        run_dpic_test(args)
