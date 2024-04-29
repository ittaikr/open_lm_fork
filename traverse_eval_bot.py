from open_lm.train import evaluate
from open_lm.data import get_data
from open_lm.model import create_model
from open_lm.distributed import is_master, init_distributed_device
from open_lm.main import load_model

import os
import yaml
import argparse
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import json
import re

def parse_resume_path(resume_path):
    # resume_path is flop_1.25e+16_step_2118_poly_08_1.pt or flop_1.25e+16_step_2118.pt
    # we need to extract the flop and step values, and averager (which is poly_08_1 in this case)
    # we can use the following regex to extract the values
    # flop_(\d+\.?\d*)e\+(\d+)_step_(\d+)_([a-zA-Z0-9_]+).pt
    
    regex = r"flop_(\d+\.?\d*)e\+(\d+)_step_(\d+)_([a-zA-Z0-9_]+).pt"
    match = re.match(regex, resume_path)
    if match:
        flop = float(match.group(1) + "e" + match.group(2))
        step = int(match.group(3))
        averager = match.group(4) if match.group(4) else None
        return flop, step, averager



def eval_ckpt(args, ckpt_path):
    args.resume = ckpt_path
    checkpoint_root = Path(args.resume).parent
    # checkpoint path without parent directory
    checkpoint_path_name = Path(args.resume).name
    args.train_data = None
    args.valid_data = ["/p/scratch/laionize/smyrnis1/refined_web_tokenized/{00000001..00000010}.tar"] # ~141M tokens
    args.batch_size = 16
    args.log_eval_loss = 100

    model = create_model(args)
    device = init_distributed_device(args)
    model = model.to(device)
    start_epoch = load_model(args, model, None)

    data = get_data(
        args,
        epoch=start_epoch, # the value of start_epoch doesn't matter here, but we keep the same code from main.py
        tokenizer=None,
        skip_train=args.dataset_metadata is not None,
    )

    metrics = evaluate(model, data, start_epoch, args, None)
    metrics["checkpoint_path"] = args.resume
    metrics["val_data"] = args.val_data
    metrics["model"] = args.model
    metrics["flop"], metrics["step"], metrics["averager"] = parse_resume_path(checkpoint_path_name)

    path_dir_to_save = os.path.join(Path(checkpoint_root).parent, "eval_results")
    if not os.path.exists(path_dir_to_save):
        os.makedirs(path_dir_to_save)

    if is_master(args):
        with open(os.path.join(path_dir_to_save, checkpoint_path_name + "_eval_result.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

def get_args(exp_path):
    parser = argparse.ArgumentParser(description='Eval Config', add_help=False)
    for sub_dir in os.listdir(exp_path):
        sub_dir_path = os.path.join(exp_path, sub_dir)
        if "args" in sub_dir_path:
            args = parser.parse_args()
            with open(sub_dir_path, 'r') as f:
                cfg = yaml.safe_load(f)
                # Override argparse defaults with config file
                for key, value in cfg.items():
                    setattr(args, key, value)
            return args

def traverse(base_path):
    for exp in os.listdir(base_path):
        exp_path = os.path.join(base_path, exp)
        if not os.path.isdir(exp_path):
            continue

        args = get_args(exp_path)

        for sub_dir in os.listdir(exp_path):     
            sub_dir_path = os.path.join(exp_path, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue
            # that means that sub_dir is the checkpoint directory
            for ckpt in os.listdir(sub_dir_path):
                ckpt_path = os.path.join(sub_dir_path, ckpt)
                if "flop" not in ckpt_path:
                    continue
                # eval_in_progress_path would be exps_sweep/24-04-28-final_sweep_cosine/000_24-04-28-final_sweep_cosine+bat_siz=1+mod=layers=15_hidden-dim=640+tra_num_sam=216924160+war_tok=216924160+lr=0.048+lr_coo_end=0.00048/checkpoints/epoch_10_eval_in_progress
                eval_in_progress_path = ckpt_path + "_eval_in_progress"
                result_path = ckpt_path + "_eval_result.jsonl"

                # wait for a random number of seconds,1 uniform between 0.5 and 2
                time.sleep(random.uniform(0.5, 2))

                if os.path.exists(eval_in_progress_path):
                    print(f"Skipping {ckpt_path} as evaluation is in progress")
                    continue
                
                if os.path.exists(result_path):
                    print(f"Skipping {ckpt_path} as evaluation is already done")
                    continue
                
                # create eval_in_progress file
                with open(eval_in_progress_path, 'w') as f:
                    f.write("")
                try:
                    print(f"Starting evaluation for {ckpt_path}")
                    # eval_ckpt(args, ckpt_path)
                    os.remove(eval_in_progress_path)
                except Exception as e:
                    print(f"Error in evaluating {ckpt_path}")
                    print(e)
                    os.remove(eval_in_progress_path)
                    continue
                
                # eval_ckpt(args, ckpt_path)
                print(f"Starting evaluation for {ckpt_path}")
                # return


if __name__ == "__main__":
    while True:
        dirs_to_traverse = [os.path.join("exps_final_runs", exp) for exp in os.listdir("exps_final_runs") if os.path.isdir(os.path.join("exps_final_runs", exp))]
        for dir in dirs_to_traverse:
            traverse(dir)