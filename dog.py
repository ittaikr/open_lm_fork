import pdb

from torch.optim import Optimizer
import torch
from torch.autograd import grad
import logging

import numpy as np

required = 'REQUIRED'

class DoG(Optimizer):

    @torch.no_grad()
    def __init__(self, params, lr=required,
                 init_dist_abs=0.0,
                 init_dist_rel=required,
                 init_dist_rel_normalized=None,
                 granularity='all',
                 weight_decay=0.0,
                 decay_to_init=False,
                 eps=1e-8,
                 decouple_decay: bool = False,
                 decay_factor: float = None,
                 max_rbar: bool = False,
                 normalize_rbar: bool = False,
                 moving_avg_init: str = None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if init_dist_abs is not required and init_dist_abs < 0.0:
            raise ValueError("Invalid init_dist: {}".format(lr))
        if init_dist_rel is not required and init_dist_rel < 0.0:
            raise ValueError("Invalid init_dist: {}".format(lr))
        if init_dist_rel_normalized is not None and init_dist_rel_normalized < 0.0:
            raise ValueError("Invalid init_dist: {}".format(init_dist_rel_normalized))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if init_dist_rel_normalized is None:
            init_dist_rel_normalized = init_dist_rel

        self._step = 0
        self.granularity = granularity

        defaults = dict(lr=lr, init_dist_abs=init_dist_abs, init_dist_rel=init_dist_rel, init_dist_rel_normalized=init_dist_rel_normalized,
                        granularity=granularity, weight_decay=weight_decay, eps=eps,
                        decay_to_init=decay_to_init, decouple_decay=decouple_decay, decay_factor=decay_factor, weight_decay_factor=[1.0] * len(params), max_rbar=max_rbar, normalize_rbar=normalize_rbar, gm=1.0, moving_avg_init=moving_avg_init)
        super().__init__(params, defaults)

        with torch.no_grad():
            for group in self.param_groups:
                for p in zip(group['params']):
                    if hasattr(p, 'to_normalize') and p.to_normalize:
                        p /= torch.norm_except_dim(p, 2, 0)

        for group in self.param_groups:
            self._update_dim(group)
            self._init_weight_norm(group)

    def __setstate__(self, state):
        super().__setstate__(state)
        # todo: make sure state is loaded correctly

    @torch.no_grad()
    def get_stats(self):
        if self.granularity == 'param':
            init_dist = []
            eta = []
            for group in self.param_groups:
                init_dist.append(np.array([d.norm().cpu() for d in group['d']]))
                eta.append(np.array([(eta.sum()/eta.numel()).cpu() for eta in group['eta']]))
            return {"dog/init_dist": np.concatenate(init_dist),
                    "dog/eta": np.concatenate(eta)}
        for group in self.param_groups:
                return {"dog/init_dist": group['d'],
                    "dog/eta": group['eta'][0].item()}

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        first_step = self._step == 0

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            decay_to_init = group['decay_to_init']
            decouple_decay = group['decouple_decay']

            granularity = group['granularity']

            if first_step:
                group['init_buffer'] = [torch.clone(p).detach() for p in group['params']]
                group['init_tag_buffer'] = [p if (group['moving_avg_init'] is None) else torch.clone(p).detach() for p in group['init_buffer']]
            init = group['init_buffer']
            init_tag = group['init_tag_buffer']

            if weight_decay > 0 and not decouple_decay:
                for p, pi in zip(group['params'], init):
                    if p.grad is None or (hasattr(p, 'to_normalize') and p.to_normalize):
                        continue
                    if decay_to_init:
                        if not first_step:
                            p.grad.add_(p - pi, alpha=weight_decay)
                    else:
                        p.grad.add_(p, alpha=weight_decay)

            if granularity == 'all':  # treat all layers as one long vector
                if first_step:
                    curr_d = group['d'] = group['init_dist_rel'] + \
                                 group['init_dist_rel'] * torch.stack([p.norm() for p in group['params']]).norm().item()
                    curr_d_decay = 0.0
                    group['g2sum'] = 0.0
                else:
                    # curr_d = np.linalg.norm([torch.norm(p - pi).item() for p, pi in
                    #                          zip(group['params'], init)])
                    curr_d = curr_d_decay = torch.stack([torch.norm(p.detach() - pi) for p, pi in
                                           zip(group['params'], init)]).norm().item()
                    group['d'] = max(group['d'], curr_d)

                if not (group['decay_factor'] is None):
                    curr_norm = torch.stack([torch.norm(p.detach()) for p, pi in
                                                zip(group['params'], init)]).norm().item()
                    group['weight_decay_factor'] = [min( group['decay_factor'] * group['d'] / curr_norm, 1.0)] * len(group['params'])
                
                # group['g2'] = np.sum([(p.grad ** 2).sum().item() for p in group['params']])
                group['g2'] = torch.stack([((p.grad.detach() * group['gm']) ** 2).sum() for p in group['params']]).sum().item()
                group['g2sum'] = group['g2sum'] + group['g2']
                
                group['eta'] = [group['lr'] * group['d'] / (np.sqrt(group['g2sum']) + group['eps'])] * len(group['params'])

            elif granularity == 'param':  # treat each param in the group as a separate block
                if first_step:
                    group['d'] = [None] * len(group['params'])
                    group['min_d'] = [None] * len(group['params'])
                    group['g2'] = [None] * len(group['params'])
                    group['g2sum'] = [None] * len(group['params'])
                    group['g2sum_forget'] = [None] * len(group['params'])
                    group['eta'] = [None] * len(group['params'])

                for i in range(len(group['params'])):
                    p, pi = group['params'][i], init_tag[i]
                    norm_dim = group['dim'][i]

                    if first_step:
                        group['d'][i] = group['init_dist_rel'] * torch.ones(1)[0] +\
                                    ( group['init_dist_rel'] if not (hasattr(p, 'to_normalize') and p.to_normalize) else group['init_dist_rel_normalized'] ) * self._norm(p, dim=norm_dim)
                        group['g2sum'][i] = torch.zeros_like(group['d'][i])
                        group['g2sum_forget'][i] = torch.zeros_like(group['d'][i])

                        logging.info( "dims: " + str(p.shape) + "," + str(group['d'][i].shape) )
                        if hasattr(p, 'per_inout'):
                            logging.info( "per_inout: " + p.per_inout)

                        if group['max_rbar'] or group['normalize_rbar']:
                            group['d'][i] = group['d'][i].max()

                        group['min_d'][i] = torch.clone(group['d'][i]).detach()
                    else:
                        curr_d = self._norm(p - pi, dim=norm_dim)
                        if group['max_rbar']:
                            curr_d = curr_d.max()
                        elif group['normalize_rbar']:
                            curr_d = torch.norm(curr_d) / (curr_d.numel() ** 0.5)
                        group['d'][i] = torch.maximum(group['d'][i], group['min_d'][i])
                        group['d'][i] = torch.maximum(group['d'][i], curr_d)

                    if not (group['decay_factor'] is None):
                        curr_norm = self._norm(p, dim=norm_dim)
                        group['weight_decay_factor'][i] = torch.minimum( group['decay_factor'] * group['d'][i] / curr_norm, torch.ones_like(curr_norm))
                    
                    group['g2'][i] = self._sum((p.grad * group['gm']) ** 2, dim=norm_dim)
                    group['g2sum'][i].add_(group['g2'][i])

                    group['eta'][i] = group['lr'] * group['d'][i] / (torch.sqrt(group['g2sum'][i] - group['g2sum_forget'][i]) + group['eps'])

            elif granularity == 'element':  # like diagonal adagrad - not tested!
                raise NotImplementedError('the init distant for element is not yet implemented correctly')
    
            else:
                logging.error(f'Unknown granularity {granularity}')


            if weight_decay > 0 and decouple_decay:
                for p, pi, wdf in zip(group['params'], init, group['weight_decay_factor']):
                    if p.grad is None or (hasattr(p, 'to_normalize') and p.to_normalize):
                        continue
                    if decay_to_init:
                        if not first_step:
                            p.add_(p - pi, alpha=-weight_decay)
                    else:
                        pv = p
                        if hasattr(p, 'num_of_heads'):
                            pv = p.view(p.num_of_heads, -1)
                        pv.add_(pv * wdf, alpha=-weight_decay)

            # pdb.set_trace()
            for p, eta in zip(group['params'], group['eta']):
                if p.grad is None:
                    continue
                pv, p_grad = p, p.grad
                if hasattr(p, 'num_of_heads'):
                    pv, p_grad = pv.view(p.num_of_heads, -1), p_grad.view(p.num_of_heads, -1)
                pv.add_(p_grad * group['gm'] * eta, alpha=-1.0)
                    
            for p in group['params']:
                if hasattr(p, 'to_normalize') and p.to_normalize:
                    p /= torch.norm_except_dim(p, 2, 0)

            self._normalize_weight(group)

            self._update_avg_if_needed(group)

                # # implementing weight decay as composite term - STUPID!!!
                # if group['weight_decay'] > 0.0:
                #     p /= (1 + eta * weight_decay)

        # # log
        # if first_step:
        #     logging.info(f'Number of param groups: {len(self.param_groups)}')
        #     for i, group in enumerate(self.param_groups, 1):
        #         logging.info(f'Group {i}: {len(group["params"])} params')
        # # report = {}
        # for i, group in enumerate(self.param_groups, 1):
        #     eta_min, eta_max, eta_mean = min(group['eta']), max(group['eta']), np.mean(group['eta'])
        #     # logging.info(f'Group {i}: {len(group["params"])} params, eta_min={eta_min:.3g}, eta_max={eta_max:.3g}, eta_mean={eta_mean:.3g}')
        #     # report.update({f"g{i}/eta_min": eta_min, f"g{i}/eta_max": eta_max, f"g{i}/eta_mean": eta_mean})
        # # if _is_comet:
        # #     comet_ml.get_global_experiment().log_metrics(report, step=self._step)


        self._step += 1

        return loss

    def reset(self):
        self._step = 0

    def _update_dim(self, group):
        assert self._step == 0

        group['dim'] = [None] * len(group['params'])
        group['norm_dim'] = [None] * len(group['params'])

        for i, p  in enumerate(group['params']):
            if hasattr(p, 'num_of_heads'):
                group['dim'][i] = p.num_of_heads
                group['norm_dim'][i] = p.num_of_heads
            else:
                norm_dim = [j for j in range(p.dim())]
                if hasattr(p, 'per_inout'):
                    if p.per_inout == 'first' or p.per_inout == 'output':
                        norm_dim.remove(0)
                    elif p.per_inout == 'last' or p.per_inout == 'input':
                        norm_dim.remove(p.dim()-1)
                    elif p.per_inout == 'all':
                        norm_dim = []
                group['dim'][i] = tuple(norm_dim)
                if (not hasattr(p, 'per_inout')) or p.per_inout != 'all':
                    group['norm_dim'][i] = group['dim'][i]
                else:
                    norm_dim = [j for j in range(p.dim())]
                    if hasattr(p, 'norm_type'):
                        if p.norm_type == 'first' or p.norm_type == 'output':
                            norm_dim.remove(0)
                        elif p.norm_type == 'last' or p.norm_type == 'input':
                            norm_dim.remove(p.dim()-1)
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

            if self._step == 0:
                logging.info( "normalize factor: " + str(x_factor) )

    def _update_avg_if_needed(self, group):
        for i in range(len(group['params'])):
            p = group['params'][i]
            init_tag = group['init_tag_buffer'][i]
            g2sum_forget = group['g2sum_forget'][i]
            g2sum = group['g2sum'][i]

            if group['moving_avg_init'] is None or group['moving_avg_init'] == 'none':
                return

            method, *args = group['moving_avg_init'].split('_')
            if method == "poly":
                poly_n = 0.0 if not args else float(args[0])
                freq = 1 if (len(args) < 2 or not args) else int(args[1])
                t = self._step + 2

                if t % freq != 0:
                    return

                alpha = ((poly_n + 1) / (poly_n + t))
            if method == 'ema' or method == 'ema2':
                alpha = 1 - (0.99 if not args else float(args[0]))
            else:
                raise Exception("Not Implemented")

            init_tag.mul_(1-alpha).add_(p, alpha=alpha)
            group['d'][i].mul_(1-alpha)
            if method != 'ema2':
                g2sum_forget.mul_(1-alpha).add_(group['g2sum'][i], alpha=alpha)
            else:
                g2sum.mul_(1-alpha)

            

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