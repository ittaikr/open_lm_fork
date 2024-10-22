import numpy as np


def assign_learning_rate(optimizer, new_lr, gradient_scheduler=False):
    for param_group in optimizer.param_groups:
        if gradient_scheduler:
            param_group["gm"] = new_lr
        else:
            param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step, warmup_start=0):
    return warmup_start + (base_lr-warmup_start) * (step + 1) / warmup_length

def _cosine_lr(step, base_lr, warmup_length, steps, min_lr, force_min_lr):
    if step < warmup_length:
        lr = _warmup_lr(base_lr, warmup_length, step)
    else:
        e = step - warmup_length
        es = steps - warmup_length
        lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr)
        lr = max(lr, force_min_lr)
    return lr

def const_lr(optimizer, base_lr, warmup_length):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def const_lr_cooldown(optimizer, base_lr, warmup_length, steps, cooldown_steps, cooldown_power=1.0, cooldown_end_lr=0.):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e/es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps, min_lr, force_min_lr, warmup_start=0, gradient_scheduler=False):
    if gradient_scheduler:
        base_lr = 1.
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step, warmup_start=warmup_start)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr)
            lr = max(lr, force_min_lr)
        assign_learning_rate(optimizer, lr, gradient_scheduler)
        return lr
    return _lr_adjuster

def cosine_rewarmed_lr(optimizer, base_lr, warmup_length, steps, min_lr, force_min_lr, target_steps, original_warmup):
    def _lr_adjuster(step):
        new_base_lr = _cosine_lr(target_steps - steps + warmup_length, base_lr, original_warmup, target_steps, min_lr, force_min_lr)
        if step < warmup_length:
            lr = _warmup_lr(new_base_lr, warmup_length, step)
        else:
            lr = _cosine_lr(target_steps - steps + step - warmup_length, base_lr, warmup_length, target_steps - warmup_length, min_lr, force_min_lr)
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def hybrid_cosine_rsqrt(optimizer, base_lr, warmup_length, steps, min_lr, d = 2.0):
    def _lr_adjuster(step):
        es = steps - warmup_length
        C = (1 + np.cos(np.pi / d)) / 2
        beta = (min_lr / (base_lr - min_lr) + C) * es / (np.sin(np.pi / d) * np.pi) - (es / d)
        alpha = ((base_lr - min_lr) * C - min_lr) * np.sqrt(beta + es / d)
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        elif step <= steps / d:
            e = step - warmup_length
            # es = steps - warmup_length
            lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr)
        else:
            e = step - warmup_length
            lr = alpha / np.sqrt(e + beta)
        lr_ret = max(lr, min_lr)
        assign_learning_rate(optimizer, lr_ret)
        return lr_ret
    return _lr_adjuster

def hybrid_cosine_rsqrt_cooldown(optimizer, base_lr, warmup_length, steps, min_lr, cooldown_steps, d = 2.0, cooldown_power=1.0, cooldown_end_lr=0.):
    def _lr_adjuster(step):
        es = steps - warmup_length
        C = (1 + np.cos(np.pi / d)) / 2
        beta = (min_lr / (base_lr - min_lr) + C) * es / (np.sin(np.pi / d) * np.pi) - (es / d)
        alpha = ((base_lr - min_lr) * C - min_lr) * np.sqrt(beta + es / d)
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        elif step <= steps / d:
            e = step - warmup_length
            # es = steps - warmup_length
            lr = min_lr + 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - min_lr)
        elif step < start_cooldown_step:
            e = step - warmup_length
            lr = alpha / np.sqrt(e + beta)
        else:
            e = step - start_cooldown_step
            es = steps - start_cooldown_step
            # linear decay if power == 1; polynomial decay otherwise;
            decay = (1 - (e/es)) ** cooldown_power
            lr = decay * (alpha / np.sqrt(start_cooldown_step + beta)) + cooldown_end_lr
        lr_ret = max(lr, min_lr)
        assign_learning_rate(optimizer, lr_ret)
        return lr_ret
    return _lr_adjuster