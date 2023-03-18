'''
Author: Yen-Ju Chen  mru.11@nycu.edu.tw
Date: 2023-03-10 10:58:15
LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
LastEditTime: 2023-03-18 11:44:33
FilePath: /mru/Knowledge-Distillation/utils.py
Description: 

'''
import os
import yaml
import shutil
from datetime import datetime
from termcolor import colored, cprint
from torch.utils.tensorboard import SummaryWriter


def log_file(log_root: str, data: dict, fname=None):

    # named file name as datetime
    if fname:
        log_dir = os.path.join(log_root, fname)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_root, timestamp)
    
    # make directory
    if not os.path.isdir("./logs"):
        os.mkdir("./logs")
    if not os.path.exists(log_root):
        os.mkdir(log_root)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    elif fname in ["testing", "test"] and os.path.isdir(log_dir):
        print("Removing the original test / testing file...")
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)
    else:
        print("Directory already exist, change a file name...")
        os._exit(1)
    
    # save parameters
    with open(os.path.join(log_dir, "args.yaml"), 'w') as f:
        yaml.dump(data, f, Dumper=yaml.CDumper)

    return log_dir


def logger(text, title=False, log_dir=None):

    # stdout
    if title:
        cprint(f"\n=========================== {text} ===========================", "green")
    else:
        cprint(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] \t {text}", "blue")

    # file
    if title:
        with open('{}/train_record.txt'.format(log_dir), 'a') as train_record:
            train_record.write(f'\n=========================== {text} ===========================\n')
    else:
        with open('{}/train_record.txt'.format(log_dir), 'a') as train_record:
            train_record.write(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] \t {text}\n")



