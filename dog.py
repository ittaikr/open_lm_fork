import pdb

from torch.optim import Optimizer
import torch
from torch.autograd import grad
import logging

import numpy as np

required = 'REQUIRED'

class DoG(Optimizer):
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
                 normalize_rbar: bool = False):
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

        defaults = dict(lr=lr, init_dist_abs=init_dist_abs, init_dist_rel=init_dist_rel, init_dist_rel_normalized=init_dist_rel_normalized,
                        granularity=granularity, weight_decay=weight_decay, eps=eps,
                        decay_to_init=decay_to_init, decouple_decay=decouple_decay, decay_factor=decay_factor, weight_decay_factor=[1.0] * len(params), max_rbar=max_rbar, normalize_rbar=normalize_rbar)
        super().__init__(params, defaults)

        with torch.no_grad():
            for group in self.param_groups:
                for p in zip(group['params']):
                    if hasattr(p, 'to_normalize') and p.to_normalize:
                        p /= torch.norm_except_dim(p, 2, 0)

    def __setstate__(self, state):
        super().__setstate__(state)
        # todo: make sure state is loaded correctly

    @torch.no_grad()
    def get_stats(self):
        for group in self.param_groups:
            if group['granularity'] == 'param':
                return {"dog/init_dist": np.array([d.norm().cpu() for d in group['d']]),
                    "dog/eta": np.array([(eta.sum()/eta.numel()).cpu() for eta in group['eta']])}
            else:
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

            if granularity == 'all':  # treat all layers as one long vector
                if first_step:
                    curr_d = group['d'] = group['init_dist_abs'] + \
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
                group['g2'] = torch.stack([(p.grad.detach() ** 2).sum() for p in group['params']]).sum().item()
                group['g2sum'] = group['g2sum'] + group['g2']
                
                group['eta'] = [group['lr'] * group['d'] / np.sqrt(group['g2sum'])] * len(group['params'])

            elif granularity == 'param':  # treat each param in the group as a separate block
                if first_step:
                    group['d'] = [torch.zeros(1)[0]] * len(group['params'])
                    group['g2'] = [torch.zeros(1)[0]] * len(group['params'])
                    group['g2sum'] = [torch.zeros(1)[0]] * len(group['params'])
                    group['eta'] = [torch.zeros(1)[0]] * len(group['params'])

                for i in range(len(group['params'])):
                    p, pi = group['params'][i], init[i]
                    norm_dim = [j for j in range(p.dim())]
                    if hasattr(p, 'per_inout'):
                        if p.per_inout == 'first' or p.per_inout == 'output':
                            norm_dim.remove(0)
                        elif p.per_inout == 'last' or p.per_inout == 'input':
                            norm_dim.remove(p.dim()-1)
                        elif p.per_inout == 'all':
                            norm_dim = []
                    norm_dim = tuple(norm_dim) 

                    if first_step:
                        group['d'][i] = group['init_dist_abs'] * torch.ones(1)[0] +\
                                    ( group['init_dist_rel'] if not (hasattr(p, 'to_normalize') and p.to_normalize) else group['init_dist_rel_normalized'] ) * self._norm(p, dim=norm_dim)
                        group['g2sum'][i] = group['eps'] * torch.ones_like(group['d'][i])

                        logging.info( "dims: " + str(p.shape) + "," + str(group['d'][i].shape) )
                        if hasattr(p, 'per_inout'):
                            logging.info( "per_inout: " + p.per_inout)

                        if group['max_rbar'] or group['normalize_rbar']:
                            group['d'][i] = group['d'][i].max()
                    else:
                        curr_d = self._norm(p - pi, dim=norm_dim)
                        if group['max_rbar']:
                            curr_d = curr_d.max()
                        elif group['normalize_rbar']:
                            curr_d = torch.norm(curr_d) / (curr_d.numel() ** 0.5)
                        group['d'][i] = torch.maximum(group['d'][i], curr_d)

                    if not (group['decay_factor'] is None):
                        curr_norm = self._norm(p, dim=norm_dim)
                        group['weight_decay_factor'][i] = torch.minimum( group['decay_factor'] * group['d'][i] / curr_norm, torch.ones_like(curr_norm))
                    
                    group['g2'][i] = self._sum(p.grad ** 2, dim=norm_dim)
                    group['g2sum'][i] = group['g2sum'][i] + group['g2'][i]

                    group['eta'][i] = group['lr'] * group['d'][i] / torch.sqrt(group['g2sum'][i])

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
                        p.add_(p * wdf, alpha=-weight_decay)

            # pdb.set_trace()
            for p, eta in zip(group['params'], group['eta']):
                if p.grad is None:
                    continue
                p.add_(p.grad * eta, alpha=-1.0)
                    
            for p in group['params']:
                if hasattr(p, 'to_normalize') and p.to_normalize:
                    p /= torch.norm_except_dim(p, 2, 0)

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

    def _sum(self, p, dim):
        if dim == tuple():
            return p
        return p.sum(dim=dim, keepdim=True)

    def _norm(self, p, dim):
        if dim == tuple():
            return p
        return p.norm(dim=dim, keepdim=True)