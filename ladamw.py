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

    @torch.no_grad()
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, g_op=None, quantile=None):

        if lr < 0.0:
            raise ValueError(f'Invalid learning rate ({lr}). Suggested value is 1.')

        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')

        self.t = 0
        self._first_step = True

        defaults = dict(lr=lr, beta1 = betas[0], beta2 = betas[1], eps=eps, weight_decay=weight_decay, g_op=g_op, quantile=quantile)
        super(LAdamW, self).__init__(params, defaults)

        for group in self.param_groups:
            self._update_dim(group)
            self._init_weight_norm(group)

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
                if g_op == 'max' or g_op == 'max_m' or g_op == 'max_g':
                    group['v'] = [self._amax(p.clone().detach_(), dim=group['dim'][i]).zero_() for i,p in enumerate(group['params'])]
                else:
                    group['v'] = [p.clone().detach_().zero_() for p in group['params']]

            for i, (x, m, v) in enumerate(zip(group['params'], group['m'], group['v'])):
                if x.grad is None:
                    continue
                else:
                    m.mul_(beta1)
                    m.add_(x.grad, alpha=1-beta1)
                    m_hat = m / (1-beta1 ** self.t)
                    m_hat_v = m_hat
                    xv = x
                    if hasattr(x, 'num_of_heads'):
                        m_hat_v = m_hat_v.view(x.num_of_heads, -1)
                        xv = xv.view(x.num_of_heads, -1)

                    if not (g_op == 'max_m' or g_op == 'max_g'):
                        v.mul_(beta2)
                        if g_op == 'max':
                            v.add_(self._amax(x.grad**2, dim=group['dim'][i], quantile=group['quantile']), alpha=1-beta2)
                            v_hat_dim = tuple()
                        else:
                            v.add_(x.grad**2, alpha=1-beta2)
                            v_hat_dim = group['dim'][i]

                        if g_op == 'max_tag':
                            v_hat = self._amax(v, dim=v_hat_dim, quantile=group['quantile']) / (1-beta2 ** self.t)
                        else:
                            v_hat = self._sum(v,dim=v_hat_dim) / (1-beta2 ** self.t)
                        if g_op == 'avg':
                            v_hat /= group['numel'][i]
                    elif g_op == 'max_m':
                        v_hat = self._amax(m_hat**2, dim=group['dim'][i], quantile=group['quantile'])
                    elif g_op == 'max_g':
                        v_hat = self._amax(x.grad**2, dim=group['dim'][i], quantile=group['quantile'])

                    if self._first_step:
                         logging.info( "shapes: " + str((v.shape, v_hat.shape, group['dim'][i])) )

                    if (g_op != 'avg') and not (group['quantile'] is None):
                        quantile_m = self._amax(m_hat**2, dim=group['dim'][i], quantile=group['quantile'])
                        if self._first_step:
                            logging.info( "shapes m_hat: " + str((m_hat_v.shape, quantile_m.shape, group['dim'][i])) )
                        m_hat_v = m_hat_v.where(m_hat_v <= quantile_m, quantile_m)

                    xv.add_( m_hat_v / (v_hat.sqrt() + eps) , alpha= -lr )

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

    def _update_dim(self, group):
        assert self._first_step

        group['dim'] = [None for i in range(len(group['params']))]
        group['norm_dim'] = [None] * len(group['params'])
        group['numel'] = [None for i in range(len(group['params']))]
        for i, x  in enumerate(group['params']):
            if hasattr(x, 'num_of_heads'):
                group['dim'][i] = x.num_of_heads
                group['norm_dim'][i] = x.num_of_heads
                group['numel'][i] = x.numel() / x.num_of_heads
            else:
                norm_dim = [j for j in range(x.dim())]
                numel = x.numel()
                if hasattr(x, 'per_inout'):
                    if x.per_inout == 'first' or x.per_inout == 'output':
                        norm_dim.remove(0)
                        numel /= x.shape[0]
                    elif x.per_inout == 'last' or x.per_inout == 'input':
                        norm_dim.remove(x.dim()-1)
                        numel /= x.shape[x.dim()-1]
                    elif x.per_inout == 'all':
                        norm_dim = []
                        numel = 1
                group['dim'][i] = tuple(norm_dim)
                group['numel'][i] = numel
                if (not hasattr(x, 'per_inout')) or x.per_inout != 'all':
                        group['norm_dim'][i] = group['dim'][i]
                else:
                    norm_dim = [j for j in range(x.dim())]
                    if hasattr(x, 'norm_type'):
                        if x.norm_type == 'first' or x.norm_type == 'output':
                            norm_dim.remove(0)
                        elif x.norm_type == 'last' or x.norm_type == 'input':
                            norm_dim.remove(x.dim()-1)
                    group['norm_dim'][i] = tuple(norm_dim)

    def _init_weight_norm(self, group):
        group['init_x_norm'] = [ None for p in group['params']]
        for i, x in enumerate(group['params']):
            if not (hasattr(x, 'constant_norm') and x.constant_norm):
                continue

            group['init_x_norm'][i] = self._norm(x,dim=group['norm_dim'][i]).mean()
            logging.info( "norm: " + str(group['init_x_norm'][i]) )

        self._normalize_weight(group)

    def _normalize_weight(self, group):
        for i, x in enumerate(group['params']):
            if not (hasattr(x, 'constant_norm') and x.constant_norm):
                continue

            xv = x
            if hasattr(x, 'num_of_heads'):
                xv = x.view(x.num_of_heads, -1)

            x_norm = self._norm(x,dim=group['norm_dim'][i])

            x_factor = group['init_x_norm'][i] / x_norm
            x_factor = x_factor.where(x_factor <= 1., 1.)

            xv.mul_( x_factor )

            if self._first_step:
                logging.info( "normalize factor: " + str(x_factor) )

    def _sum(self, p, dim):
        if dim == tuple():
            return p
        if isinstance(dim, tuple):
            return p.sum(dim=dim, keepdim=True)
        return p.view(dim, -1).sum(dim=-1, keepdim=True)

    def _norm(self, p, dim):
        if dim == tuple():
            return p
        if isinstance(dim, tuple):
            return p.norm(dim=dim, keepdim=True)
        return p.view(dim, -1).norm(dim=-1, keepdim=True)

    def _amax(self, p, dim, quantile=None):
        if dim == tuple():
            return p

        if isinstance(dim, tuple):
            if quantile==None:
                return p.amax(dim=dim, keepdim=True)
            assert(len(dim) == 1 or len(dim)==len(p.shape))
            if len(dim)==len(p.shape):
                dim=None
            else:
                dim = dim[0]
            return p.quantile(quantile, dim=dim, keepdim=True)

        if quantile==None:
            return p.view(dim, -1).amax(dim=-1, keepdim=True)
        return p.view(dim, -1).quantile(quantile, dim=-1, keepdim=True)
