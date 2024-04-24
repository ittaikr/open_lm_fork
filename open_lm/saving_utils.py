import logging
import os
import torch

def save_checkpoint_step(args, model, completed_flop, epoch, averagers):

    checkpoint_dict_model = {
        "epoch": epoch,
        "flop": completed_flop,
        "name": args.name,
        "state_dict": model.state_dict(),
    }

    # if there will be more than args.max_checkpoints_flops checkpoints, remove the oldest one
    # first, count the number of files that have "flop_" in their name
    flop_file_counter = 0
    for file in os.listdir(args.checkpoint_path):
        if "flop_" in file:
            flop_file_counter += 1
    
    if os.path.exists(os.path.join(args.checkpoint_path, f"flop_{completed_flop:.2e}.pt")):
        return # in case of resuming, we don't want to save the same checkpoint twice
    if flop_file_counter >= args.max_checkpoints_flops:
        oldest_flop = min([float(file.split("_")[1].split(".")[0]) for file in os.listdir(args.checkpoint_path) if "flop_" in file])
        # remove all files that have the oldest flop in their name, including averagers
        for file in os.listdir(args.checkpoint_path):
            if f"_{oldest_flop:.2e}" in file:
                os.remove(os.path.join(args.checkpoint_path, file))


    torch.save(
            checkpoint_dict_model,
            os.path.join(args.checkpoint_path, f"flop_{completed_flop:.2e}.pt"),
        )
    if averagers is not None:
        for k in averagers.avgs_dict:
            torch.save(
                averagers.avgs_dict[k].get_state_dict_avg(),
                os.path.join(args.checkpoint_path, f"{k}_{completed_flop:.2e}.pt"),
            )