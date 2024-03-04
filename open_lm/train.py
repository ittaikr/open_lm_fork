import ast
import json
import logging
import math
import os
import time
import csv
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.autograd import profiler

try:
    import wandb
except ImportError:
    wandb = None

from .distributed import is_master
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(
    model, data, loss, epoch, optimizer, scaler, scheduler, averagers, args, tb_writer=None, csv_path=None
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()

    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = AverageMeter()
    if averagers is not None and args.log_avg_model_training_loss:
        losses_avg_m = {key: AverageMeter() for key in averagers.avgs_dict.keys()}
        local_avg_losses = {}
        total_loss_avg = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()

    # used only if --log-logit-mean flag is passed
    logit_m = AverageMeter()

    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        if not args.skip_scheduler:
            scheduler(step)

        (texts,) = batch
        texts = torch.LongTensor(texts).to(device)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # with profiler.profile(profile_memory=True, use_cuda=True) as prof:
                inputs = texts[:, : args.seq_len - 1]
                targets = texts[:, 1 : args.seq_len]
                out, _ = model(inputs)
                # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
                if args.log_logit_mean:
                    logit_m.update(torch.mean(out).item())
                total_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))
                    
                if averagers is not None and args.log_avg_model_training_loss and i % args.log_avg_model_training_loss == 0:
                    for key, averager in averagers.avgs_dict.items():
                        with torch.no_grad():
                            out_avg, _ = averager.av_model(inputs)
                            # save the loss for the average model for logging on cpu
                            total_loss_avg[key] = loss(out_avg.reshape(-1, args.vocab_size), targets.reshape(-1))
                        # averager.av_model.cpu()
                
            backward(total_loss, scaler)
            
        else:
            # split up batch into accum_freq chunks -- if you have --batch-size 8 and --accum-freq 4
            # then you only process 2 items at a time. batch-size must be divisible by accume-freq.
            assert (
                args.batch_size % args.accum_freq == 0
            ), "Batch size must be divisible by accum_freq"
            per_batch = args.batch_size // args.accum_freq
            inputs = texts[:, : args.seq_len - 1]
            targets = texts[:, 1 : args.seq_len]
            for ii in range(args.accum_freq):
                with autocast():
                    inputs_ii = inputs[ii * per_batch : (ii + 1) * per_batch]
                    targets_ii = targets[ii * per_batch : (ii + 1) * per_batch]
                    out, _ = model(inputs_ii)

                    if args.log_logit_mean:
                        logit_m.update(torch.mean(out).item())

                    local_loss = (
                        loss(out.reshape(-1, args.vocab_size), targets_ii.reshape(-1))
                        / args.accum_freq
                    )
                    # calc avergers loss every args.log_avg_model_training_loss steps
                    if averagers is not None and args.log_avg_model_training_loss and i % args.log_avg_model_training_loss == 0:
                        for key, averager in averagers.avgs_dict.items():
                            with torch.no_grad():
                                out_avg, _ = averager.av_model(inputs_ii).item()
                                local_avg_losses[key] = loss(out_avg.reshape(-1, args.vocab_size), targets.reshape(-1)) / args.accum_freq
                            # averager.av_model.cpu()
                backward(local_loss, scaler)
                if ii == 0:
                    total_loss = local_loss
                    if averagers is not None and args.log_avg_model_training_loss and i % args.log_avg_model_training_loss == 0:
                        for key, averager in averagers.avgs_dict.items():
                            total_loss_avg[key] = local_avg_losses[key]
                else:
                    total_loss += local_loss
                    if averagers is not None and args.log_avg_model_training_loss and i % args.log_avg_model_training_loss == 0:
                        for key, averager in averagers.avgs_dict.items():
                            total_loss_avg[key] += local_avg_losses[key]

        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(args.grad_clip_norm, norm_type=2.0)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
            optimizer.step()
        if averagers is not None:
            averagers.step()
        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        samples_per_second = inputs.numel() * args.world_size / batch_time_m.val
        samples_per_second_per_gpu = inputs.numel() / batch_time_m.val
        if i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch:
            batch_size = len(inputs)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # Aggregate loss across all processes
            gathered_loss = [torch.zeros_like(total_loss) for _ in range(args.world_size)]
            torch.distributed.all_gather(gathered_loss, total_loss)
            avg_loss = sum(gathered_loss).item() / args.world_size
            losses_m.update(avg_loss, batch_size * args.world_size)
            
            gathered_loss_avg = {}
            if averagers is not None and args.log_avg_model_training_loss and (i % args.log_avg_model_training_loss == 0 or batch_count == num_batches_per_epoch):
                for key in total_loss_avg.keys():
                    gathered_loss_avg[key] = [torch.zeros_like(total_loss_avg[key]) for _ in range(args.world_size)]
                    torch.distributed.all_gather(gathered_loss_avg[key], total_loss_avg[key])
                    avg_loss_avg = sum(gathered_loss_avg[key]).item() / args.world_size
                    losses_avg_m[key].update(avg_loss_avg, batch_size * args.world_size)

            # Ensure averagers and other metrics are similarly aggregated
            # Example for samples_per_second (assuming it needs to be aggregated)
            # Use all_reduce for in-place aggregation
            sps_tensor = torch.tensor(samples_per_second).to(total_loss.device)
            torch.distributed.all_reduce(sps_tensor, op=torch.distributed.ReduceOp.SUM)
            samples_per_second = sps_tensor.item() / args.world_size 
            sps_gpu_tensor = torch.tensor(samples_per_second_per_gpu).to(device)
            torch.distributed.all_reduce(sps_gpu_tensor, op=torch.distributed.ReduceOp.SUM)
            samples_per_second_per_gpu_avg = sps_gpu_tensor.item() / args.world_size
            if is_master(args):
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {losses_m.avg:.3f} "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu_avg:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "loss": losses_m.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": optimizer.param_groups[0]["lr"],
                    "tokens": (step + 1) * args.batch_size * args.seq_len * args.world_size,
                }
                if averagers is not None and args.log_avg_model_training_loss:
                    for k in averagers.avgs_dict:
                        if averagers is not None and args.log_avg_model_training_loss and (i % args.log_avg_model_training_loss == 0 or batch_count == num_batches_per_epoch):
                            log_data[k + "_loss"] = losses_avg_m[k].val
                if args.log_logit_mean:
                    log_data["logit_mean"] = logit_m.val

                for name, val in log_data.items():
                    name = "train/" + name
                    if tb_writer is not None:
                        tb_writer.add_scalar(name, val, step)
                    if args.wandb:
                        assert wandb is not None, "Please install wandb."
                        wandb.log({name: val, "step": step, "tokens": log_data["tokens"]})
                if csv_path is not None:
                    # if the file does not exist, (which is the case for the first iteration) we need to write the header
                    rowd = OrderedDict(epoch=epoch, step=step)
                    rowd.update([("train/" + k, v) for k, v in log_data.items()])
                    if not os.path.exists(csv_path):
                        with open(csv_path, "w") as f:
                            dict_writer = csv.DictWriter(f, fieldnames=rowd.keys())
                            dict_writer.writeheader()
                    # delete all rows with epoch <= current epoch
                    with open(csv_path, "a") as f:
                        dict_writer = csv.DictWriter(f, fieldnames=rowd.keys())
                        dict_writer.writerow(rowd)


            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, start_epoch, args, writer):
    """
    evaluates perplexity on validation data
    """
    if is_master(args):
        print("=> begin evaluation")
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.eval()

    data["val"].set_epoch(
        start_epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["val"].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    sps_m = AverageMeter()
    spspg_m = AverageMeter()
    end = time.time()
    loss = torch.nn.CrossEntropyLoss()
    for i, batch in enumerate(dataloader):
        (texts,) = batch
        texts = torch.LongTensor(texts).to(device)

        data_time_m.update(time.time() - end)

        with autocast():
            inputs = texts[:, : args.seq_len - 1]
            targets = texts[:, 1 : args.seq_len]
            out, _ = model(inputs)
            total_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))
            losses_m.update(total_loss.item(), inputs.shape[0])
        batch_time_m.update(time.time() - end)
        sps_m.update(inputs.numel() * args.world_size / batch_time_m.val)
        spspg_m.update(inputs.numel() / batch_time_m.val)

    # Save eval loss / etc.
    log_data = {
        "loss": losses_m.avg,
        "data_time": data_time_m.avg,
        "batch_time": batch_time_m.avg,
        "samples_per_second": sps_m.avg,
        "samples_per_second_per_gpu": spspg_m.avg,
        "tokens": start_epoch * args.train_num_samples * args.seq_len,
    }

    for name, val in log_data.items():
        name = "valid/" + name
        if writer is not None:
            writer.add_scalar(name, val, start_epoch)
        if args.wandb and is_master(args):
            assert wandb is not None, "Please install wandb."
            wandb.log({name: val, "epoch": start_epoch, "tokens": log_data["tokens"]})
    if is_master(args):
        print(f"evaluation perplexity: {math.exp(losses_m.avg)}")
    return log_data
