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
    # we need to extract the flop and step values, and averager (which is poly_08_1 in the first case)
    # do that with splitting the string by _ and then parsing the values

    # get the averager part, if it exists
    if "poly" in resume_path:
        # averager = resume_path.split("_")[-3] + "_" + resume_path.split("_")[-2] + "_" + resume_path.split("_")[-1].split(".")[0]
        match = re.search(r"poly_\d+_\d+", resume_path)
        averager = match.group(0) if match else None
    else:
        averager = None
    
    # get the flop and step values
    flop = resume_path.split("_")[1]
    step = resume_path.split("_")[3] if averager else resume_path.split("_")[3].split(".")[0]

    return float(flop), int(step), averager



def eval_ckpt(args, ckpt_path):
    args.resume = ckpt_path
    checkpoint_root = Path(args.resume).parent
    # checkpoint path without parent directory
    checkpoint_path_name = Path(args.resume).name
    args.train_data = None
    if args.data_key == "json.gz":
        args.val_data = ["/p/fastdata/mmlaion/lmdata_2/refined_web_tokenized/{00000001..00000010}.tar"]# ~141M tokens
        if "openwebtext2" in args.dataset_manifest:
            args.val_data = ["/p/fastdata/mmlaion/lmdata_2/openwebtext2_tokenized/{0000001..00000013}.tar"]
    elif args.data_key == "json":
        args.val_data = ["/p/fastdata/mmlaion/lmdata/rpj/shard_{00000000..00000003}.tar"]
    args.ignore_parse_errors = False
    args.dataset_manifest = None
    args.batch_size = 16
    args.log_eval_loss = 50
    args.wandb = False
    model = create_model(args)
    device = init_distributed_device(args)
    model = model.to(device)
    start_epoch, global_step, pretrained_seed = load_model(args, model, None)

    data = get_data(
        args,
        epoch=0, # the value of start_epoch doesn't matter here, but we keep the same code from main.py
        tokenizer=None,
        skip_train=True,
    )

    metrics = evaluate(model, data, 0, args, None)
    metrics["checkpoint_path"] = args.resume
    metrics["val_data"] = args.val_data
    metrics["model"] = args.model
    if 'flop' in args.resume:
        metrics["flop"], metrics["step"], metrics["averager"] = parse_resume_path(checkpoint_path_name)
    else:
        metrics["epoch"] = args.resume.split("_")[-1].split(".")[0]
        # metrics["averager"] = None if "poly" not in args.resume else args.resume.split("_")[-3] + "_" + args.resume.split("_")[-2] + "_" + args.resume.split("_")[-1].split(".")[0]
        match = re.search(r"poly_\d+_\d+", args.resume)
        metrics["averager"] = match.group(0) if match else None
        metrics["step"] = global_step
    path_dir_to_save = os.path.join(Path(checkpoint_root).parent, "eval_results")
    if not os.path.exists(path_dir_to_save):
        os.makedirs(path_dir_to_save)

    if is_master(args):
        with open(os.path.join(path_dir_to_save, checkpoint_path_name + "_eval_result.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

def get_args(exp_path):
    parser = argparse.ArgumentParser(description='Eval Config', add_help=False)
    for file in os.listdir(exp_path):
        file_path = os.path.join(exp_path, file)
        if "args" in file_path:
            args = parser.parse_args()
            with open(file_path, 'r') as f:
                cfg = yaml.safe_load(f)
                # Override argparse defaults with config file
                for key, value in cfg.items():
                    setattr(args, key, value)
            return args

def traverse(base_path, skip_not_flop=False):
    for exp in os.listdir(base_path):
        exp_path = os.path.join(base_path, exp)
        if not os.path.isdir(exp_path): # skip job.yaml file
            continue
        if '30-' in exp:
            continue
        args = get_args(exp_path)
        for sub_dir in os.listdir(exp_path):     
            sub_dir_path = os.path.join(exp_path, sub_dir)
            if not os.path.isdir(sub_dir_path) or 'results' in sub_dir_path:
                continue
            # that means that sub_dir is the checkpoint directory
            for ckpt in os.listdir(sub_dir_path):
                ckpt_path = os.path.join(sub_dir_path, ckpt)
                if "optimizer" in ckpt_path: # for now evaluate only the checkpoints that have flop in their name
                    continue

                if "eval_in_progress" in ckpt_path or "results.jsonl" in ckpt_path:
                    continue

                if skip_not_flop and "flop" not in ckpt_path:
                    continue
                
                eval_in_progress_path = ckpt_path + "_eval_in_progress"
                result_path = os.path.join(os.path.join(exp_path, "eval_results"), ckpt + "_eval_result.jsonl")

                time.sleep(random.uniform(0.5, 2))

                # continue if evaluation is in progress
                # that is, if eval_in_progress file exists for less then 4 hours or result file exists
                skip = False
                if os.path.exists(eval_in_progress_path):
                    if (time.time() - os.path.getmtime(eval_in_progress_path)) < 4 * 60 * 60:
                        skip = True
                
                if os.path.exists(result_path):
                    skip = True
                    
                if skip:
                    continue
                # create eval_in_progress file
                with open(eval_in_progress_path, 'w') as f:
                    f.write("")
                try:
                    print(f"Starting evaluation for {ckpt_path}")
                    eval_ckpt(args, ckpt_path)
                    os.remove(eval_in_progress_path)
                except Exception as e:
                    print(f"Error in evaluating {ckpt_path}")
                    print(e)
                    os.remove(eval_in_progress_path)
                    continue


if __name__ == "__main__":
    flop_dir_to_traverse = "exps_final_runs"
    sweep_dir = "exps_sweep"
    original_dir = "exps"
    while True:
        dirs_to_traverse = [os.path.join(flop_dir_to_traverse, exp) for exp in os.listdir(flop_dir_to_traverse) if os.path.isdir(os.path.join(flop_dir_to_traverse, exp))]
        for dir in dirs_to_traverse:
            # if '24-05-09' not in dir:
            #     continue
            traverse(dir, skip_not_flop=False)
        print(f"Finished {flop_dir_to_traverse}")
        # dirs_to_traverse = [os.path.join(sweep_dir, exp) for exp in os.listdir(sweep_dir) if os.path.isdir(os.path.join(sweep_dir, exp))]
        # for dir in dirs_to_traverse:
        #     traverse(dir)
        # print("Finished sweep")
        