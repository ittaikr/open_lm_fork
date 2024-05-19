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
                 phase_start_factor=None,
                 discount_p=0.0,
                 proximal_reg=False,
                 decay_to_init=False,
                 theta_1=1.0,
                 theta_2=0.0,
                 safe_lr='none',
                 eps=1e-8,
                 decouple_decay: bool = False,
                 decay_factor: float = None):
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
                        phase_start_factor=phase_start_factor,
                        theta_1=theta_1, theta_2=theta_2,
                        discount_p=discount_p,
                        decay_to_init=decay_to_init, decouple_decay=decouple_decay, decay_factor=decay_factor, weight_decay_factor=[1.0] * len(params),
                        proximal_reg=proximal_reg, safe_lr=safe_lr)
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
                return {"dog/init_dist": np.array(group['d']),
                    "dog/eta": np.array(group['eta'])}
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

            theta_1, theta_2 = group['theta_1'], group['theta_2']

            granularity = group['granularity']

            if group['phase_start_factor'] is not None:
                safe_lr_coef = np.pi / 2 / (1 - 2 * group['phase_start_factor'] ** (-2))
            else:
                safe_lr_coef = np.pi / 2

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
                    prev_d = curr_d = group['phase_d'] = group['d'] = group['init_dist_abs'] + \
                                 group['init_dist_rel'] * torch.stack([p.norm() for p in group['params']]).norm().item()
                    curr_d_decay = 0.0
                    group['g2sum'] = 0.0
                    group['g2sum_ratio'] = 1.0
                else:
                    # curr_d = np.linalg.norm([torch.norm(p - pi).item() for p, pi in
                    #                          zip(group['params'], init)])
                    curr_d = curr_d_decay = torch.stack([torch.norm(p.detach() - pi) for p, pi in
                                           zip(group['params'], init)]).norm().item()
                    group['d'], prev_d = max(group['d'], curr_d), group['d']

                    if group['phase_start_factor'] is not None and group['d'] > group['phase_start_factor'] * group['phase_d']:
                        logging.info('Starting a new DoG phase')
                        group['phase_d'] = group['d']
                        group['g2sum'] = 0.0
                        group['g2sum_ratio'] = 1.0

                if not (group['decay_factor'] is None):
                    curr_norm = torch.stack([torch.norm(p.detach()) for p, pi in
                                                zip(group['params'], init)]).norm().item()
                    group['weight_decay_factor'] = [min( group['decay_factor'] * curr_d_decay / curr_norm, 1.0)] * len(group['params'])
                
                # group['g2'] = np.sum([(p.grad ** 2).sum().item() for p in group['params']])
                group['g2'] = torch.stack([(p.grad.detach() ** 2).sum() for p in group['params']]).sum().item()
                if group['safe_lr'] == 'g2':
                    g2_multiplier = (safe_lr_coef * np.log(group['g2sum_ratio'])) ** 2 + 1
                else:
                    g2_multiplier = 1.0
                prev_g2sum = group['g2sum']
                if group['discount_p'] == 0.0:
                    group['g2sum'] = group['g2sum'] + group['g2'] * g2_multiplier
                else:
                    group['g2sum'] = ((prev_d / curr_d) ** (group['discount_p'])) * group['g2sum'] + group['g2'] * g2_multiplier
                group['g2sum_ratio'] *= group['g2sum'] / max(prev_g2sum, group['g2'])  # the max takes care of the special case of phase start
                if group['safe_lr'] == 'g2sum':
                    g2sum_multiplier = (safe_lr_coef * np.log(group['g2sum_ratio'])) ** 2 + 1
                else:
                    g2sum_multiplier = 1.0
                group['g2sum_'] = (group['g2sum'] + (theta_1 ** 2 - 1) * group['g2']) * g2sum_multiplier
                
                group['eta'] = [group['lr'] * group['d'] / np.sqrt(group['g2sum_'])] * len(group['params'])
                group['lambda'] = [theta_2 * group['g2'] / (group['d'] * np.sqrt(group['g2sum_']))] * len(group['params'])

            elif granularity == 'param':  # treat each param in the group as a separate block
                if first_step:
                    prev_d = curr_d = group['d'] = group['init_dist_abs'] * torch.ones(len(group['params'])) +\
                                torch.tensor([ ( group['init_dist_rel'] if not (hasattr(p, 'to_normalize') and p.to_normalize) else group['init_dist_rel_normalized'] ) * p.norm() for p in group['params']])
                    group['phase_d'] = group['d'].clone().detach()
                    group['g2sum'] = group['eps'] * torch.ones_like(group['d'])
                    group['g2sum_ratio'] = torch.ones_like(group['d'])
                else:
                    curr_d = torch.tensor([torch.norm(p - pi) for p, pi in
                                           zip(group['params'], init)])
                    group['d'], prev_d = torch.maximum(group['d'], curr_d), group['d']
                    if group['phase_start_factor'] is not None:
                        new_phase_mask = group['d'] > group['phase_start_factor'] * group['phase_d']
                        if torch.any(new_phase_mask):
                            logging.info(f'Starting new phases for {new_phase_mask.sum()} parameters, with indexes {torch.where(new_phase_mask)[0].numpy()}')
                            group['g2sum'][new_phase_mask] = 0.0
                            group['g2sum_ratio'][new_phase_mask] = 1.0
                            group['phase_d'][new_phase_mask] = group['d'][new_phase_mask]

                if not (group['decay_factor'] is None):
                    curr_norm = torch.tensor([torch.norm(p) for p, pi in
                                           zip(group['params'], init)])
                    group['weight_decay_factor'] = torch.minimum( group['decay_factor'] * curr_d / curr_norm, torch.ones(len(curr_norm)))
                
                group['g2'] = torch.tensor([(p.grad ** 2).sum() for p in group['params']])
                if group['safe_lr'] == 'g2':
                    g2_multiplier = (safe_lr_coef * torch.log(group['g2sum_ratio'])) ** 2 + 1
                else:
                    g2_multiplier = 1.0

                prev_g2sum = group['g2sum']
                if group['discount_p'] == 0.0:
                    group['g2sum'] = group['g2sum'] + group['g2'] * g2_multiplier
                else:
                    group['g2sum'] = ((prev_d / curr_d) ** (group['discount_p'])) * group['g2sum'] + group['g2'] * g2_multiplier
                group['g2sum_ratio'] *= group['g2sum'] / torch.maximum(prev_g2sum, group['g2'])  # the max takes care of the special case of phase start

                if group['safe_lr'] == 'g2sum':
                    g2sum_multiplier = (safe_lr_coef * torch.log(group['g2sum_ratio'])) ** 2 + 1
                else:
                    g2sum_multiplier = 1.0
                group['g2sum_'] = (group['g2sum'] + (theta_1 ** 2 - 1) * group['g2']) * g2sum_multiplier

                group['eta'] = list(group['lr'] * group['d'] / torch.sqrt(group['g2sum_']))
                group['lambda'] = list(theta_2 * group['g2'] / (group['d'] * np.sqrt(group['g2sum_'])))

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
                        p.add_(p, alpha=-weight_decay * wdf)

            # pdb.set_trace()
            if theta_2 == 0.0:
                for p, eta in zip(group['params'], group['eta']):
                    if p.grad is None:
                        continue
                    p.add_(p.grad, alpha=-eta)
            elif group['proximal_reg']:
                for p, pi, eta, lam in zip(group['params'], group['init_buffer'], group['eta'], group['lambda']):
                    if p.grad is None:
                        continue
                    p.div_(1 + lam * eta)
                    p.add_(pi, alpha=(lam * eta/(1 + lam * eta)))
                    p.add_(p.grad, alpha=-eta/(1 + lam * eta))
            else:
                for p, pi, eta, lam in zip(group['params'], group['init_buffer'], group['eta'], group['lambda']):
                    if p.grad is None:
                        continue
                    p.add_(p.grad + lam * (p - pi), alpha=-eta)
                    
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


class ScalarAdagrad(Optimizer):
    def __init__(self, params, lr=required,
                 granularity='all', weight_decay=0.0, eps=1e-16):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.first_step = True

        defaults = dict(lr=lr, granularity=granularity,
                        weight_decay=weight_decay, eps=eps)
        super(ScalarAdagrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ScalarAdagrad, self).__setstate__(state)
        # todo: make sure state is loaded correctly

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

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            granularity = group['granularity']

            if granularity == 'all':
                if self.first_step:
                    group['g2'] = np.sum([(p.grad ** 2).sum().item() for p in group['params']])
                else:
                    group['g2'] += np.sum([(p.grad ** 2).sum().item() for p in group['params']])
                group['eta'] = [group['lr'] / np.sqrt(group['g2'] + group['eps'])] * len(group['params'])
            elif granularity == 'param':
                if self.first_step:
                    group['g2'] = torch.Tensor([(p.grad ** 2).sum().item() for p in group['params']])
                else:
                    group['g2'] += torch.Tensor([(p.grad ** 2).sum().item() for p in group['params']])
                group['eta'] = list(group['lr'] / torch.sqrt(group['g2'] + group['eps']))
            # elif granularity == 'element':
            #     if first_step:
            #         group['d'] = [group['init_dist'] * torch.ones_like(p) for p in group['params']]
            #         group['g2'] = [(p.grad ** 2) + group['eps'] for p in group['params']]
            #     else:
            #         curr_d = [torch.abs(p-pi) for p, pi in zip(group['params'], init)]
            #         group['d'] = [torch.maximum(d, cd) for d, cd in zip(group['d'], curr_d)]
            #         group['g2'] = [g2 + (p.grad ** 2) for g2, p in zip(group['g2'], group['params'])]
            #     group['eta'] = [group['lr'] * d / torch.sqrt(g2) for d, g2 in zip(group['d'], group['g2'])]
            else:
                logging.error(f'Unknown granularity {granularity}')

            self.first_step = False

            for p, eta in zip(group['params'], group['eta']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                #     if nesterov:
                #         d_p = d_p.add(buf, alpha=momentum)
                #     else:
                #         d_p = buf
                if granularity == 'element':
                    p.add_(-eta * p.grad)
                else:
                    p.add_(p.grad, alpha=-eta)

                # implementing weight decay as composite term
                if group['weight_decay'] > 0.0:
                    p /= (1 + eta * weight_decay)

        return loss

    def reset(self):
        self.first_step = True


class CutkoskyOrabona18(Optimizer):
    def __init__(self, params, lr=1.0, gn=1.0, weight_decay=0.0, eps=1e-8):
        if weight_decay != 0.0:
            logging.error('CutkoskyOrabona18 implementation does not currently support non-zero weight decay (use l2_reg instead)')
            raise ValueError()

        if len(params) > 1:
            logging.error('CutkoskyOrabona18 implementation does not currently support more than a single vector parameter')
            raise ValueError()

        if torch.norm(params[0]) > 0.0:
            logging.error('CutkoskyOrabona18 implementation does not currently support non-zero initialization')
            raise ValueError()

        self.first_step = True

        self.t = 0
        self.coef = 2 / (2 - np.log(3))

        defaults = dict(lr=lr, gn=gn, weight_decay=weight_decay, eps=eps)
        super(CutkoskyOrabona18, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CutkoskyOrabona18, self).__setstate__(state)
        # todo: make sure state is loaded correctly

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

        self.t += 1

        group = self.param_groups[0]
        gn, eps, param = group['gn'], group['eps'], group['params'][0]

        if self.first_step:
            wealth = torch.Tensor([eps])[0]
            v = torch.zeros(1)[0]
            y = torch.zeros_like(param, requires_grad=False)
            g2 = torch.zeros(1)[0]
            A = torch.ones(1)[0]
            self.first_step = False
        else:
            wealth, v, y, g2, A = (group[k] for k in ['wealth', 'v', 'y', 'g2', 'A'])

        grad = param.grad

        # ONS step
        s = grad.view(-1) @ y.view(-1) / gn
        if s.abs().item() > 1.0:
            logging.info(f'Clipping occurred at CutkoskyOrabona18 at step {self.t}')
            s = s.clip(-1.0, 1.0)
        w = v * wealth
        wealth -= s * w
        z = s / (1 - s * v)
        A += z ** 2
        v = (v - self.coef * z / A).clip(-0.5, 0.5)

        # Scalar Adagrad step
        g2 += grad.view(-1) @ grad.view(-1)
        y -= group['lr'] * grad / torch.sqrt(g2)
        y /= torch.maximum(y.norm(), torch.ones(1)[0])

        # Combination
        param.data[:] = gn * w * y

        group['wealth'] = wealth
        group['v'] = v
        group['y'] = y
        group['g2'] = g2
        group['A'] = A
        group['s'] = s

        return loss

    def reset(self):
        self.first_step = True
        self.t = 0


class CG(Optimizer):
    r"""
    Exact conjugate gradient implementation using Hessian-vector products.
    Note: for quadratic optimization there is a 2x more efficient solution
    because actually we can always compute the next gradient from the HVP with
    the updated momentum.
    """

    def __init__(self, params, weight_decay=0.0):
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(weight_decay=weight_decay, lr=0.0)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    # @torch.no_grad()
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

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if weight_decay != 0:  # 95% sure this implements WD correctly
                    g = g.add(p, alpha=weight_decay)
                gg = g.view(-1) @ g.view(-1)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    mom = param_state['momentum_buffer'] = torch.clone(g).detach()
                else:
                    mom = param_state['momentum_buffer']
                    gg_prev = param_state['gg_prev']
                    rho = gg / gg_prev
                    mom.mul_(rho.item()).add_(g.detach())
                param_state['gg_prev'] = gg

                hvp = grad(g.view(-1) @ mom.view(-1), p)[0]
                eta = gg / (mom.view(-1) @ hvp.view(-1))

                p.data.add_(mom, alpha=-eta.item())

        return loss


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss

    def reset(self):
        self.first_step = True
