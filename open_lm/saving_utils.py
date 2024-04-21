import logging
import os
import torch

def save_checkpoint_step(args, model, completed_flop, averagers):

    checkpoint_dict_model = {
        "epoch": completed_flop,
        "name": args.name,
        "state_dict": model.state_dict(),
    }

    torch.save(
            checkpoint_dict_model,
            os.path.join(args.checkpoint_path, f"epoch_{completed_flop:.2e}.pt"),
        )
    if averagers is not None:
        for k in averagers.avgs_dict:
            # logging.info('=> saving averager for {}'.format(k))
            torch.save(
                averagers.avgs_dict[k].get_state_dict_avg(),
                os.path.join(args.checkpoint_path, f"{k}_{completed_flop:.2e}.pt"),
            )