import ast
import json
import logging
import math
import os
import time
from csv import DictWriter
from collections import OrderedDict
import torch.distributed as dist
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.autograd import profiler
from open_lm.data import sample_chunk
try:
    import wandb
except ImportError:
    wandb = None

from .distributed import is_master
from .precision import get_autocast
#import schedulefree

from .saving_utils import save_checkpoint_step

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

def log_avg(args, step):
    return args.log_avg_model_training_loss and (step % args.log_avg_model_training_loss == 0)

def log_model(args, step, i, has_data):
    if i == 0:
        return False
    elif has_data < args.world_size:
        return True
    else:
        return step % args.log_every_n_steps == 0

def train_one_epoch(
    model, data, loss, epoch, step, optimizer, scaler, scheduler, total_steps, averagers, args, tb_writer=None, csv_path=None
):
    start_step = step
    # for saving checkpoints based on flops

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()
    if args.schedulefree:
        optimizer.train()
    
    data["train"].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches

    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    if args.schedulefree:
        losses_schedfree = 0
        losses_schedfree_m = AverageMeter()
    losses_m = AverageMeter()
    losses_m_ret = AverageMeter()
    if args.z_loss_coefficient != 0.0:
        z_losses_m = AverageMeter()
        z_losses_m_ret = AverageMeter()
    if averagers is not None and args.log_avg_model_training_loss:
        losses_avg_m = {key: AverageMeter() for key in averagers.avgs_dict.keys()}
        local_avg_losses = {}
        total_loss_avg = {}
        local_tuple_loss_avg = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    batch_time_m_ret = AverageMeter()
    data_time_m_ret = AverageMeter()
    grad_norm = AverageMeter()
    grad_norm_ret = AverageMeter()
    grads_norms_ret = AverageMeter()
    grad_clipped = 0
    grad_clipped_ret = 0

    # used only if --log-logit-mean flag is passed
    logit_m = AverageMeter()
    logit_m_ret = AverageMeter()

    end = time.time()

    data_iterator = iter(dataloader)
    batch_count = 0

    if args.max_tokens is not None:
        total_steps = min(total_steps, args.max_tokens // (args.batch_size * args.world_size) + 1)

    for i in itertools.count():

        # step = num_batches_per_epoch * epoch + i
        if not args.skip_scheduler:
            scheduler(step)

        if step >= total_steps:
            logging.warning(f"step: {step} has reached/exceeded total_steps: {total_steps}. ending training.")
            break

        try:
            batch = next(data_iterator)
            has_data = torch.tensor(1, dtype=torch.long, device=device)
        except StopIteration:
            has_data = torch.tensor(0, dtype=torch.long, device=device)

        if args.world_size > 1:
            dist.all_reduce(has_data, op=ReduceOp.SUM)
        else:
            has_data = has_data.item()

        if i==1:
            batch_size = len(inputs)
            logging.info(f'batch_size: {batch_size}')

        # start of logging
        if is_master(args) and log_model(args, step, i, has_data):
            batch_size = len(inputs)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            samples_per_second = inputs.numel() * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = inputs.numel() / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {losses_m.avg:.3f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": losses_m.avg,
                "data_time": data_time_m.avg,
                "batch_time": batch_time_m.avg,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"],
                "tokens": (step) * args.batch_size * args.seq_len * args.world_size,
                "grad_clipped": grad_clipped,
                "grad_norm": grad_norm.avg,
            }
            if args.z_loss_coefficient != 0.0:
                log_data["z_loss"] = z_losses_m.avg
            if log_avg(args, step):
                if averagers is not None:
                    for key, value in losses_avg_m.items():
                        log_data[key + "_loss"] = value.avg
                if args.schedulefree:
                    log_data["schedfree_loss"] = losses_schedfree_m.avg
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
                        dict_writer = DictWriter(f, fieldnames=rowd.keys())
                        dict_writer.writeheader()
                # delete all rows with epoch <= current epoch
                with open(csv_path, "a") as f:
                    dict_writer = DictWriter(f, fieldnames=rowd.keys())
                    dict_writer.writerow(rowd)


            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
            # reset all average meters
            losses_m.reset()
            if args.z_loss_coefficient != 0.0:
                z_losses_m.reset()
            if averagers is not None and log_avg(args, step):
                for k in averagers.avgs_dict.keys():
                    losses_avg_m[k].reset()
            if args.schedulefree:
                losses_schedfree_m.reset()
            grad_norm.reset()
            grad_clipped = 0
        # end of logging

        if has_data < args.world_size:
            break

        (texts,) = batch
        texts = torch.LongTensor(texts).to(device)
        data_time_m.update(time.time() - end)
        data_time_m_ret.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                inputs, targets = sample_chunk(texts, args)
                out, _ = model(inputs)
                if args.log_logit_mean:
                    logit_m.update(torch.mean(out).item())
                    logit_m_ret.update(torch.mean(out).item())
                tuple_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))
                total_loss = tuple_loss[1] if args.z_loss_coefficient != 0.0 else tuple_loss
                total_zloss = tuple_loss[0] if args.z_loss_coefficient != 0.0 else None
            backward(tuple_loss[0] if args.z_loss_coefficient != 0.0 else tuple_loss, scaler)
            if log_avg(args, step+1):
                if averagers is not None:
                    with autocast():
                        inputs = texts[:, : args.seq_len - 1]
                        targets = texts[:, 1 : args.seq_len]
                        for key, averager in averagers.avgs_dict.items():
                            with torch.no_grad():
                                out_avg, _ = averager.av_model(inputs)
                                # save the loss for the average model for logging
                                tuple_loss_avg = loss(out_avg.reshape(-1, args.vocab_size), targets.reshape(-1))
                                total_loss_avg[key] = tuple_loss_avg[1] if args.z_loss_coefficient != 0.0 else tuple_loss_avg
                if args.schedulefree:
                    model.eval()
                    optimizer.eval()
                    with autocast():
                        with torch.no_grad():
                            out_schedfree, _ = model(inputs)
                            losses_schedfree_tuple = loss(out_schedfree.reshape(-1, args.vocab_size), targets.reshape(-1))
                            losses_schedfree = losses_schedfree_tuple[1] if args.z_loss_coefficient != 0.0 else losses_schedfree_tuple
                    model.train()
                    optimizer.train()
            
        else:
            assert (
                args.batch_size % args.accum_freq == 0
            ), "Batch size must be divisible by accum_freq"
            per_batch = args.batch_size // args.accum_freq
            inputs, targets = sample_chunk(texts, args)
            for ii in range(args.accum_freq):
                with autocast():
                    inputs_ii = inputs[ii * per_batch : (ii + 1) * per_batch]
                    if inputs_ii.shape[0] == 0:
                        break
                    targets_ii = targets[ii * per_batch : (ii + 1) * per_batch]
                    out, _ = model(inputs_ii)

                    if args.log_logit_mean:
                        logit_m.update(torch.mean(out).item())
                    local_tuple_loss = (
                        loss(out.reshape(-1, args.vocab_size), targets_ii.reshape(-1))
                    )
                    local_loss = local_tuple_loss[1] if args.z_loss_coefficient != 0.0 else local_tuple_loss
                    local_loss /= args.accum_freq
                    local_zloss = local_tuple_loss[0] if args.z_loss_coefficient != 0.0 else None
                    local_zloss /= args.accum_freq
                backward(local_tuple_loss[0] if args.z_loss_coefficient != 0.0 else local_tuple_loss, scaler)
                with autocast():
                    inputs_ii = inputs[ii * per_batch : (ii + 1) * per_batch]
                    if inputs_ii.shape[0] == 0:
                        break
                    targets_ii = targets[ii * per_batch : (ii + 1) * per_batch]
                    if log_avg(args, step+1):
                        if averagers is not None:
                            for key, averager in averagers.avgs_dict.items():
                                with torch.no_grad():
                                    out_avg, _ = averager.av_model(inputs_ii)
                                    local_tuple_loss_avg[key] = loss(out_avg.reshape(-1, args.vocab_size), targets_ii.reshape(-1))
                                    local_avg_losses[key] = local_tuple_loss_avg[key][1] if args.z_loss_coefficient != 0.0 else local_tuple_loss_avg[key]
                                    local_avg_losses[key] /= args.accum_freq
                        if args.schedulefree:
                            model.eval()
                            optimizer.eval()
                            with torch.no_grad():
                                out_schedfree, _ = model(inputs_ii)
                                local_losses_schedfree = loss(out_schedfree.reshape(-1, args.vocab_size), targets_ii.reshape(-1)) / args.accum_freq
                            model.train()
                            optimizer.train()
                if ii == 0:
                    total_loss = local_loss
                    if local_zloss is not None:
                        total_zloss = local_zloss
                    if log_avg(args, step+1):
                        if averagers is not None:
                            for key, averager in averagers.avgs_dict.items():
                                total_loss_avg[key] = local_avg_losses[key]
                        if args.schedulefree:
                            losses_schedfree = local_losses_schedfree
                else:
                    total_loss += local_loss
                    if local_zloss is not None:
                        total_zloss += local_zloss
                    if log_avg(args, step+1):
                        if averagers is not None:
                            for key, averager in averagers.avgs_dict.items():
                                total_loss_avg[key] += local_avg_losses[key]
                        if args.schedulefree:
                            losses_schedfree += local_losses_schedfree
        
        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                grad_clip_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                if isinstance(model, FSDP):
                    grad_clip_norm = model.clip_grad_norm_(args.grad_clip_norm, norm_type=2.0)
                else:
                    grad_clip_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
            optimizer.step()
        if averagers is not None:
            averagers.step()
        batch_time_m.update(time.time() - end)
        batch_time_m_ret.update(time.time() - end)
        end = time.time()

        if args.grad_clip_norm is None:
            grad_clip_norm = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm()
        grad_norm.update(grad_clip_norm.item())
        grad_norm_ret.update(grad_clip_norm.item())
        if hasattr(args, "save_grads_norms") and args.save_grads_norms:
            grads_norms_ret.update( np.array([p.grad.norm().item() for p in model.parameters() if p.requires_grad]) )

        if grad_clip_norm >= (args.fake_grad_clip_norm if args.grad_clip_norm is None else args.grad_clip_norm):
            grad_clipped += 1
            grad_clipped_ret += 1

        step += 1
        batch_count = i + 1
        # create a copy of the loss tensor to avoid modifying the original
        global_loss_tensor = total_loss.detach().clone()
        if args.z_loss_coefficient != 0.0:
            global_zloss_tensor = total_zloss.detach().clone()
        if log_avg(args, step):
            if averagers is not None:
                for key, value in total_loss_avg.items():
                    total_loss_avg[key] = value.detach().clone()
            if args.schedulefree:
                losses_schedfree = losses_schedfree.detach().clone()
        if args.world_size > 1:
            # all_reduce the copied loss tensor
            dist.all_reduce(global_loss_tensor, op=ReduceOp.AVG)
            if args.z_loss_coefficient != 0.0:
                dist.all_reduce(global_zloss_tensor, op=ReduceOp.AVG)
            if log_avg(args, step):
                if args.schedulefree:
                    dist.all_reduce(losses_schedfree, op=ReduceOp.AVG)
                if averagers is not None:
                    for key, value in total_loss_avg.items():
                        dist.all_reduce(value, op=ReduceOp.AVG)
        if is_master(args):
            batch_size = len(inputs)
            # update the loss meter with the global loss tensor every iteration
            losses_m.update(global_loss_tensor.item(), batch_size)
            losses_m_ret.update(global_loss_tensor.item(), batch_size)
            if args.z_loss_coefficient != 0.0:
                z_losses_m.update(global_zloss_tensor.item(), batch_size)
                z_losses_m_ret.update(global_zloss_tensor.item(), batch_size)
            if log_avg(args, step):
                if args.schedulefree:
                    losses_schedfree_m.update(losses_schedfree.item(), batch_size)
                if averagers is not None:
                    for key, value in total_loss_avg.items():
                        losses_avg_m[key].update(value.item(), batch_size)
                        
        if args.flops_to_save is not None and is_master(args):
            curr_flops = 6 * step * args.batch_size * args.seq_len * args.world_size * args.params_count
            prev_flops = 6 * (step - 1) * args.batch_size * args.seq_len * args.world_size * args.params_count
            if np.any( (curr_flops >= args.flops_to_save) & (prev_flops < args.flops_to_save) ):
                save_checkpoint_step(args, model, curr_flops, epoch, averagers, step)
                logging.info(f"Saved model as it reached {curr_flops} FLOPs")
        
        

        # if is_master(args) and (i % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
        #     batch_size = len(inputs)
        #     num_samples = batch_count * batch_size * args.world_size
        #     samples_per_epoch = dataloader.num_samples
        #     percent_complete = 100.0 * batch_count / num_batches_per_epoch

        #     samples_per_second = inputs.numel() * args.world_size / batch_time_m.val
        #     samples_per_second_per_gpu = inputs.numel() / batch_time_m.val
        #     logging.info(
        #         f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
        #         f"Loss: {losses_m.avg:.3f} "
        #         f"Data (t): {data_time_m.avg:.3f} "
        #         f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
        #         f"LR: {optimizer.param_groups[0]['lr']:5f} "
        #     )

        #     # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
        #     log_data = {
        #         "loss": losses_m.avg,
        #         "data_time": data_time_m.avg,
        #         "batch_time": batch_time_m.avg,
        #         "samples_per_second": samples_per_second,
        #         "samples_per_second_per_gpu": samples_per_second_per_gpu,
        #         "lr": optimizer.param_groups[0]["lr"],
        #         "tokens": (step) * args.batch_size * args.seq_len * args.world_size,
        #     }
        #     if args.z_loss_coefficient != 0.0:
        #         log_data["z_loss"] = z_losses_m.avg
        #     if log_avg(args, step):
        #         if averagers is not None:
        #             for key, value in losses_avg_m.items():
        #                 log_data[key + "_loss"] = value.avg
        #         if args.schedulefree:
        #             log_data["schedfree_loss"] = losses_schedfree_m.avg
        #     if args.log_logit_mean:
        #         log_data["logit_mean"] = logit_m.val

        #     for name, val in log_data.items():
        #         name = "train/" + name
        #         if tb_writer is not None:
        #             tb_writer.add_scalar(name, val, step)
        #         if args.wandb:
        #             assert wandb is not None, "Please install wandb."
        #             wandb.log({name: val, "step": step, "tokens": log_data["tokens"]})
        #     if csv_path is not None:
        #         # if the file does not exist, (which is the case for the first iteration) we need to write the header
        #         rowd = OrderedDict(epoch=epoch, step=step)
        #         rowd.update([("train/" + k, v) for k, v in log_data.items()])
        #         if not os.path.exists(csv_path):
        #             with open(csv_path, "w") as f:
        #                 dict_writer = DictWriter(f, fieldnames=rowd.keys())
        #                 dict_writer.writeheader()
        #         # delete all rows with epoch <= current epoch
        #         with open(csv_path, "a") as f:
        #             dict_writer = DictWriter(f, fieldnames=rowd.keys())
        #             dict_writer.writerow(rowd)


        #     # resetting batch / data time meters per log window
        #     batch_time_m.reset()
        #     data_time_m.reset()
        #     # reset all average meters
        #     losses_m.reset()
        #     if args.z_loss_coefficient != 0.0:
        #         z_losses_m.reset()
        #     if averagers is not None and log_avg(args, step):
        #         for k in averagers.avgs_dict.keys():
        #             losses_avg_m[k].reset()
        #     if args.schedulefree:
        #         losses_schedfree_m.reset()

    log_data = {
            "loss": losses_m_ret.avg,
            "data_time": data_time_m_ret.avg,
            "time": batch_time_m_ret.avg,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "tokens": (step) * args.batch_size * args.seq_len * args.world_size,
            "grad_clipped": grad_clipped_ret,
            "grad_norm": grad_norm_ret.avg,
        }
    if hasattr(args, "save_grads_norms") and args.save_grads_norms:
        log_data['grads_norms'] = grads_norms_ret.avg

    if hasattr(optimizer, 'get_stats'):
        log_data.update(optimizer.get_stats())

    return step,log_data        
    # end for

def evaluate(model, data, start_epoch, args, writer, average=None):
    """
    evaluates perplexity on validation data
    """
    if is_master(args):
        print("=> begin evaluation")

    if average is None:
        average = ""
    else:
        average = average + "/"
    
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.eval()

    data["val"].set_epoch(
        start_epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["val"].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
    metrics = {}
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

        if i==0:
            batch_size = len(texts)
            logging.info(f'batch_size: {batch_size}')

        batch_size = len(texts) // args.val_freq
        for ii in range(args.val_freq):
            with autocast():
                with torch.no_grad():
                    inputs = texts[ii * batch_size : (ii+1) * batch_size , : args.seq_len - 1]
                    targets = texts[ii * batch_size : (ii+1) * batch_size , 1 : args.seq_len]
                    out, _ = model(inputs)
                    total_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))
                    losses_m.update(total_loss.item(), inputs.shape[0])

        inputs = texts[:, : args.seq_len - 1]
        batch_time_m.update(time.time() - end)
        sps_m.update(inputs.numel() * args.world_size / batch_time_m.val)
        spspg_m.update(inputs.numel() / batch_time_m.val)
        end = time.time()
        if i % args.log_eval_loss == 0 and is_master(args):
            logging.info(
                f"Validation: [{i}/{num_batches_per_epoch}] "
                f"Loss: {losses_m.avg:.3f} "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {sps_m.avg:#g}/s, {spspg_m.avg:#g}/s/gpu "
            )
            # if i == 0:
            #     metrics["loss"] = [losses_m.avg]
            #     metrics["tokens"] = [(i + 1) * args.batch_size * args.seq_len]
            # else:
            #     metrics["loss"].append(losses_m.avg)
            #     metrics["tokens"].append((i + 1) * args.batch_size * args.seq_len)
            # # reset
            # batch_time_m.reset()
            # data_time_m.reset()
            # losses_m.reset()
            # sps_m.reset()
            # spspg_m.reset()
    print('final step is', i, 'so num of tokens in validation set is', (i + 1) * args.batch_size * args.seq_len * args.world_size)
    # Save eval loss / etc.
    log_data = {
        "loss": losses_m.avg,
        "data_time": data_time_m.avg,
        "batch_time": batch_time_m.avg,
        "samples_per_second": sps_m.avg,
        "samples_per_second_per_gpu": spspg_m.avg,
        "tokens": start_epoch * args.train_num_samples * args.seq_len,
    }

    metrics["loss"] = losses_m.avg
    metrics["tokens"] = (i + 1) * args.batch_size * args.seq_len
    metrics["data_time"] = data_time_m.avg
    metrics["batch_time"] = batch_time_m.avg

    for name, val in log_data.items():
        name = "valid/" + average + name
        if writer is not None:
            writer.add_scalar(name, val, start_epoch)
        if args.wandb and is_master(args):
            assert wandb is not None, "Please install wandb."
            wandb.log({name: val, "epoch": start_epoch, "tokens": log_data["tokens"]})
    if is_master(args):
        try:
            print(f"evaluation perplexity: {math.exp(losses_m.avg)}")
        except:
            print(f"evaluation perplexity is too heigh")
    return metrics
