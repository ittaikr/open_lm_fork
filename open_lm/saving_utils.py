import logging
import os
import torch
import re


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

def get_step(args, file_name):
    match = re.search(r'step_\d+', file_name)
    if match:
        return int(match.group().split("_")[-1])
    # full_path = os.path.join(args.checkpoint_path, file_name)
    # sd_with_mmap = torch.load(full_path, mmap=True, map_location="cpu")
    # return sd_with_mmap["step"]

def save_checkpoint_step(args, model, completed_flop, epoch, averagers, current_step):
    if os.path.exists(os.path.join(args.checkpoint_path, f"flop_{completed_flop:.2e}_step_{current_step}.pt")):
        return # in case of resuming, we don't want to save the same checkpoint twice
    flop_file_counter = 0
    for file in os.listdir(args.checkpoint_path):
        if "flop_" in file and "progress" not in file:
            flop_file_counter += 1
    num_files_to_save = 1
    if averagers is not None:
        num_files_to_save += len(averagers.avgs_dict.keys())
    if flop_file_counter >= args.max_checkpoints_flops * num_files_to_save:
        oldest_step = min([get_step(args, file) for file in os.listdir(args.checkpoint_path) if "flop_" in file and "progress" not in file])
        for file in os.listdir(args.checkpoint_path):
            if get_step(args, file) == oldest_step:
                os.remove(os.path.join(args.checkpoint_path, file))
    
    checkpoint_dict_model = {
        "epoch": epoch,
        "flop": completed_flop,
        "name": args.name,
        "state_dict": unwrap_model(model).state_dict(),
        "step": current_step,
    }
    torch.save(
            checkpoint_dict_model,
            os.path.join(args.checkpoint_path, f"flop_{completed_flop:.2e}_step_{current_step}.pt"),
        )
    if averagers is not None:
        for k in averagers.avgs_dict:
            checkpoint_dict_model = {
                "epoch": epoch,
                "flop": completed_flop,
                "name": args.name,
                "state_dict": averagers.avgs_dict[k].state_dict(),
                "step": current_step,
            }
            torch.save(
                checkpoint_dict_model,
                os.path.join(args.checkpoint_path, f"flop_{completed_flop:.2e}_step_{current_step}_{k}.pt"),
            )