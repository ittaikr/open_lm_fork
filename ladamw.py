"""
PyTorch implementation of the DoG/LDoG optimizers (Ivgi et al., 2023)
"""
import logging
from typing import Optional

import torch
from torch.optim import Optimizer

import numpy as np

logger = logging.getLogger(__name__)


class LAdamW(Optimizer):
    
    __version__ = '1.0.0'

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, g_op=None):

        if lr < 0.0:
            raise ValueError(f'Invalid learning rate ({lr}). Suggested value is 1.')

        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')

        self.t = 0
        self._first_step = True

        defaults = dict(lr=lr, beta1 = betas[0], beta2 = betas[1], eps=eps, weight_decay=weight_decay, g_op=g_op)
        super(LAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LAdamW, self).__setstate__(state)

    @torch.no_grad()
    def get_stats(self):
        # TODO: temp
        rbar = []
        for group in self.param_groups:
            rbar.append(np.array( [(group['rbar'][i].norm()).item() for i in range(len(group['params']))] ))

        return dict(rbar = np.concatenate(rbar))


    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        first_step = self._first_step
        self.t += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            decouple_decay = True # group['decouple_decay']
            decay_to_init = False # group['decay_to_init']

            if first_step:
                init = group['init_buffer'] = [p.clone().detach_() for p in group['params']]
                group['m'] = [p.clone().detach_().zero_() for p in group['params']]

                # group['y_bar'] = [p.clone().detach_().zero_() for p in group['params']]
                # for y, p in zip(group['y_bar'], group['params']):
                #     y.id_number = p.id_number

                # if self.save_y:
                #     group['y'] = [p.clone().detach_().zero_() for p in group['params']]
                #     for y, p in zip(group['y'], group['params']):
                #         y.id_number = p.id_number
            else:
                init = group['init_buffer']

            if weight_decay > 0 and not decouple_decay:
                for p, pi in zip(group['params'], init):
                    if p.grad is None or (hasattr(p, 'to_normalize') and p.to_normalize):
                        continue
                    if decay_to_init:
                        if not first_step:
                            p.grad.add_(p - pi, alpha=weight_decay)
                    else:
                        p.grad.add_(p, alpha=weight_decay)


        for group in self.param_groups:
            init = group['init_buffer']
            lr = group['lr']
            eps = group['eps']
            beta1, beta2 = group['beta1'], group['beta2']
            g_op = group['g_op']

            weight_decay = group['weight_decay']
            decouple_decay = True # group['decouple_decay']
            decay_to_init = False # group['decay_to_init']

            for p, pi in zip(group['params'], init):
                if p.grad is None:
                    continue
                else:
                    if weight_decay > 0 and decouple_decay and not (hasattr(p, 'to_normalize') and p.to_normalize):
                        if decay_to_init:
                            if not first_step:
                                p.add_(p - pi, alpha=-weight_decay * lr)
                        else:
                            p.add_(p, alpha=-weight_decay * lr)

            if self._first_step:
                group['dim'] = [None for i in range(len(group['params']))]
                group['numel'] = [None for i in range(len(group['params']))]

            for i, x in enumerate(group['params']):
                if x.grad is None:
                    continue
                else:
                    if self._first_step:
                        norm_dim = [j for j in range(x.dim())]
                        numel = x.numel()
                        if hasattr(x, 'per_inout'):
                            if x.per_inout == 'first' or x.per_inout == 'output':
                                norm_dim.remove(0)
                                numel /= x.shape[0]
                            elif x.per_inout == 'last' or x.per_inout == 'input':
                                norm_dim.remove(x.dim()-1)
                                numel /= x.shape[x.dim()-1]
                            elif p.per_inout == 'all':
                                norm_dim = []
                                numel = 1
                        group['dim'][i] = tuple(norm_dim)
                        group['numel'][i] = numel
            
            if self._first_step:
                if g_op == 'max':
                    group['v'] = [self._amax(p.clone().detach_(), dim=group['dim'][i]).zero_() for i,p in enumerate(group['params'])]
                else:
                    group['v'] = [p.clone().detach_().zero_() for p in group['params']]

            for i, (x, m, v) in enumerate(zip(group['params'], group['m'], group['v'])):
                if x.grad is None:
                    continue
                else:
                    m.mul_(beta1)
                    m.add_(x.grad, alpha=1-beta1)

                    v.mul_(beta2)
                    if g_op == 'max':
                        v.add_(self._amax(x.grad**2, dim=group['dim'][i]), alpha=1-beta2)
                        v_hat_dim = tuple()
                    else:
                        v.add_(x.grad**2, alpha=1-beta2)
                        v_hat_dim = group['dim'][i]

                    v_hat = self._sum(v,dim=v_hat_dim) / (1-beta2 ** self.t)
                    if g_op == 'avg':
                        v_hat /= group['numel'][i]

                    if self._first_step:
                         logging.info( "shapes: " + str((v.shape, v_hat.shape)) )

                    x.add_( m / (v_hat + eps).sqrt() , alpha= -lr / (1-beta1 ** self.t)  )

            self._update_group_dist(group, 'x', group['params'])
            self._update_group_total_dist(group, init)

        self._first_step = False

        return loss

    def _update_group_dist(self, group, name, param):
        if self._first_step:
            group['rbar_' + name] = [torch.zeros(1)[0] for i in range(len(group['params']))]
            group['r_' + name] = [None for i in range(len(group['params']))]

        for i in range(len(group['params'])):
            p, pi = param[i], group['init_buffer'][i]
            curr_d = self._norm(p - pi, dim=group['dim'][i])
            group['rbar_' + name][i] = torch.maximum(group['rbar_' + name][i], curr_d)
            group['r_' + name][i] = curr_d


    def _update_group_total_dist(self, group, init):
        if self._first_step:
            group['rbar'] = [torch.zeros(1)[0] for i in range(len(group['params']))]
        for i in range(len(group['params'])):
            prev_rbar = group['rbar'][i]
            group['rbar'][i] = torch.maximum(group['rbar'][i], group['rbar_x'][i])

    def _sum(self, p, dim):
        if dim == tuple():
            return p
        return p.sum(dim=dim, keepdim=True)

    def _norm(self, p, dim):
        if dim == tuple():
            return p
        return p.norm(dim=dim, keepdim=True)

    def _amax(self, p, dim):
        if dim == tuple():
            return p
        return p.amax(dim=dim, keepdim=True)
