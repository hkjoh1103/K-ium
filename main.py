# %%
# import library
import argparse

from bert.train import *

# %%
# Parser generation
parser = argparse.ArgumentParser(description="BERT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str, dest='mode')
parser.add_argument('--model', default='beomi/kcbert-base', type=str, dest='model')
'''
Huggingface에서 beomi/kcbert-base 모델을 사용했습니다.
'''

parser.add_argument("--data_fn", required=True, type=str, dest="data_fn")
parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=20, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=10, type=int, dest="num_epoch")

parser.add_argument('--max_length', default=1000, type=int, dest='max_length')

config = parser.parse_args()

# %%
# main function: train or test
if __name__ == "__main__":
    if config.mode == 'train':
        train(config)
    elif config.mode == 'test':
        test(config)
    else:
        print('INVALID MODE INPUT')