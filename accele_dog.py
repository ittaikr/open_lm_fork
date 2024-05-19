"""
PyTorch implementation of the DoG/LDoG optimizers (Ivgi et al., 2023)
"""
import logging
from typing import Optional

import torch
from torch.optim import Optimizer

import numpy as np

logger = logging.getLogger(__name__)


class AcceleDoG(Optimizer):
    """
        DoG (Distance over Gradients) is a parameter-free adaptive optimizer, proposed in
         `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule` (Ivgi et al., 2023)
    """

    __version__ = '1.0.0'

    def __init__(self, params, reps_rel: float = 1e-6, lr: float = 1.0, alpha_const: float = 0.5,
                 weight_decay: float = 0.0, decay_to_init: bool = False, eps: float = 1e-8, init_eta: Optional[float] = None,
                 granularity='all', opt_ver: int = 1, alpha_ver: int = 1, step_size_ver: int = 1, momentum: float = 0.9, decouple_decay: bool = False, decay_factor: float = None):
        r"""Distance over Gradients - an adaptive stochastic optimizer.
        DoG updates parameters x_t with stochastic gradients g_t according to:
        .. math::
            \begin{aligned}
                eta_t & = \frac{ max_{i \le t}{\|x_i - x_0\|} }{ \sqrt{\sum_{i \le t }{\|g_i\|^2 + eps}} }, \\
                x_{t+1} & = x_{t} - eta_t * g_t,
            \end{aligned}
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            reps_rel (float): value to use to compute the  initial distance (r_epsilon in the paper).
                                        Namely, the first step size is given by:
                                        (reps_rel * (1+\|x_0\|)) / (\|g_0\|^2 + eps)^{1/2}  where x_0 are the initial
                                        weights of  the model (or the parameter group), and g_0 is the gradient of the
                                        first step.
                                        As discussed in the paper, this value should be small enough to ensure that the
                                        first update step will be small enough to not cause the model to diverge.
                                        Suggested value is 1e-6, unless the model uses batch-normalization,
                                        in which case the suggested value is 1e-4. (default: 1e-6)
            lr (float, optional): learning rate (referred to as c in the paper). The default value is 1.0 and changing
                                        it is not recommended.
            weight_decay (float, optional): weight decay (L2 penalty). weight_decay * x_t is added directly
                                            to the gradient (default: 0)
            eps (float, optional): epsilon used for numerical stability - added to the sum of gradients (default: 1e-8)
            init_eta (floar, optional):  if specified, this value will be used the the initial eta (i.e.
                                        first step size), and will override the value of reps_rel (default: None)
        Example:
            >>> optimizer = DoG(model.parameters(), reps_rel=1e-6)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()
        __ https://arxiv.org/pdf/2302.12022.pdf
        """

        if lr <= 0.0:
            raise ValueError(f'Invalid learning rate ({lr}). Suggested value is 1.')
        if lr != 1.0:
            logger.warning(f'We do not recommend changing the lr parameter from its default value of 1')
        if init_eta is not None:
            if init_eta <= 0:
                raise ValueError(f'Invalid value for init_eta ({init_eta})')
            logger.info(f'Ignoring reps_rel since will be explicitly set init_eta to be {init_eta} (first step size)')
            reps_rel = 0
        else:
            if reps_rel <= 0.0:
                raise ValueError(f'Invalid reps_rel value ({reps_rel}). Suggested value is 1e-6 '
                                 '(unless the model uses batch-normalization, in which case suggested value is 1e-4)')

        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')

        self._first_step = True
        self.opt_ver = opt_ver
        self.alpha_ver = alpha_ver
        self.step_size_ver = step_size_ver
        self.granularity = granularity
        self.save_y = False

        if opt_ver in [2,3]:
            alpha_const = 1.

        defaults = dict(reps_rel=reps_rel, lr=lr, alpha_const=alpha_const, weight_decay=weight_decay, decay_to_init=decay_to_init, eps=eps, init_eta=init_eta,
                            momentum=momentum, decouple_decay=decouple_decay, decay_factor=decay_factor, x_decay_factor=1.0, w_decay_factor=1.0)
        super(AcceleDoG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AcceleDoG, self).__setstate__(state)

    @torch.no_grad()
    def get_stats(self):
        if self.granularity == 'param':
            # TODO: temp
            rbar = []
            eta = []
            alpha = []
            sum_alpha_r = []
            for group in self.param_groups:
                rbar.append(group['rbar'].item())
                eta.append(group['eta_y'][0].item())
                alpha.append(group['alpha'].item())
                sum_alpha_r.append(group['sum_alpha_r'].item())

            return dict(rbar = np.array(rbar),
                eta = np.array(eta),
                alpha = np.array(alpha),
                sum_alpha_r=np.array(sum_alpha_r))
        else:
            for group in self.param_groups:
                return dict(rbar = group['rbar'].item(),
                    rbar_x = group['rbar_x'].item(),
                    rbar_y = group['rbar_y'].item(),
                    rbar_w = group['rbar_w'].item(),
                    eta = group['eta_y'][0].item(),
                    #eta_w = group['eta_w'][0].item(),
                    alpha = group['alpha'].item(),
                    sum_alpha_r=group['sum_alpha_r'].item())

    @torch.no_grad()
    def get_average_weight(self):
        y_bar = {}
        for group in self.param_groups:
            for p in group['y_bar']:
                y_bar[p.id_number] = p.clone().detach_()
        return y_bar

    @torch.no_grad()
    def get_y(self):
        y = {}
        for group in self.param_groups:
            for p in group['y']:
                y[p.id_number] = p.clone().detach_()
        return y

    @torch.no_grad()
    def get_w(self):
        w = {}
        for group in self.param_groups:
            for p in group['w']:
                w[p.id_number] = p.clone().detach_()
        return w

    @torch.no_grad()
    def get_x(self):
        x = {}
        for group in self.param_groups:
            for p in group['params']:
                x[p.id_number] = p.clone().detach_()
        return x

    @torch.no_grad()
    def get_gamma(self, idx=0):
        idx = idx % len(self.param_groups)
        for i, group in enumerate(self.param_groups):
            if not i == idx:
                continue
            alpha_r, sum_alpha_r = group['alpha'] * group['rbar'], group['sum_alpha_r']
            return alpha_r / sum_alpha_r

        assert False

    @torch.no_grad()
    def average_step(self, group):
        if self.save_y:
            for y, p in zip(group['y'], group['params']):
                y.copy_( p )

        alpha_r, sum_alpha_r = group['alpha'] * group['rbar'], group['sum_alpha_r']
        for y_bar, y in zip(group['y_bar'], group['params']):
            y_bar.mul_( 1 - (alpha_r / sum_alpha_r) )
            y_bar.add_( (alpha_r / sum_alpha_r) * y )

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

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            decouple_decay = group['decouple_decay']
            decay_to_init = group['decay_to_init']

            if first_step:
                init = group['init_buffer'] = [p.clone().detach_() for p in group['params']]
                group['w'] = [p.clone().detach_() for p in group['params']]
                for w, p in zip(group['w'], group['params']):
                    w.id_number = p.id_number

                group['y_bar'] = [p.clone().detach_().zero_() for p in group['params']]
                for y, p in zip(group['y_bar'], group['params']):
                    y.id_number = p.id_number

                if self.save_y:
                    group['y'] = [p.clone().detach_().zero_() for p in group['params']]
                    for y, p in zip(group['y'], group['params']):
                        y.id_number = p.id_number
            else:
                init = group['init_buffer']

            self._update_group_norm(group, init)

            if weight_decay > 0 and not decouple_decay:
                for p, pi in zip(group['params'], init):
                    if p.grad is None or (hasattr(p, 'to_normalize') and p.to_normalize):
                        continue
                    if decay_to_init:
                        if not first_step:
                            p.grad.add_(p - pi, alpha=weight_decay)
                    else:
                        p.grad.add_(p, alpha=weight_decay)

            self._update_group_state(group, init)
            self._override_init_eta_if_needed(group)


        for group in self.param_groups:
            init = group['init_buffer']

            alpha = group['alpha']

            for x, w, eta_y, eta_w, pi in zip(group['params'], group['w'], group['eta_y'], group['eta_w'], init):
                if x.grad is None:
                    continue
                else:
                    if weight_decay > 0 and decouple_decay and not (hasattr(x, 'to_normalize') and x.to_normalize):
                        if decay_to_init:
                            if not first_step:
                                x.add_(x - pi, alpha=-weight_decay)
                                w.add_(w - pi, alpha=-weight_decay)
                        else:
                            x.add_(x, alpha=-weight_decay * group['x_decay_factor'])
                            w.add_(w, alpha=-weight_decay * group['w_decay_factor'])

                    x.add_(x.grad.detach(), alpha=-eta_y)
                    w.add_(x.grad.detach(), alpha=-eta_w * alpha)

            self.average_step( group )

            curr_d_y = torch.stack([torch.norm(y.detach() - pi) for y, pi in zip(group['params'], init)]).norm()
            self._update_group_dist(group, 'y', group['params'])
            self._update_group_dist(group, 'w', group['w'])

            #recalculate tau
            self._update_group_total_dist(group, init)
            alpha = group['alpha']

            if self.opt_ver == 2:
                tau = group['alpha'] * group['rbar'] / group['sum_alpha_r']
            if self.opt_ver == 3:
                tau = group['alpha'] / group['sum_alpha']
            else:
                tau = 1 / alpha

            for x, w, eta_y, eta_w in zip(group['params'], group['w'], group['eta_y'], group['eta_w']):
                if x.grad is None:
                    continue
                else:
                    x.mul_(1 - tau)
                    x.add_(w, alpha=tau)

            self._update_group_dist(group, 'x', group['params'])

        self._first_step = False

        return loss

    def _update_group_dist(self, group, name, param):
        curr_d = torch.stack([torch.norm(p.detach() - pi) for p, pi in zip(param, group['init_buffer'])]).norm()
        group['rbar_' + name] = torch.maximum(group['rbar_' + name], curr_d)
        group['r_' + name] = curr_d

    def _update_group_total_dist(self, group, init):
        prev_rbar = group['rbar']
        group['rbar'] = torch.maximum(group['rbar'], group['rbar_w'])
        group['rbar_sum'] += group['rbar']

        if self.alpha_ver in [2]:
            group['alpha'] += 1
        elif self.alpha_ver in [3]:
            group['alpha'] = (1 / group['momentum']) ** torch.maximum(torch.ones(1)[0], group['alpha_const'] * group['rbar_sum']/group['rbar'] )
        elif self.alpha_ver in [4]:
            group['alpha'] *= prev_rbar / (group['rbar'] * group['momentum'])
        elif self.alpha_ver in [5]:
            group['alpha'] /= group['momentum']
        elif self.alpha_ver in [6]:
            group['alpha'] = ((1-group['momentum']) / group['momentum']) * group['sum_alpha_r'] / group['rbar']
        elif self.alpha_ver in [7]:
            pass
        else:
            group['alpha'] = torch.maximum(torch.ones(1)[0], group['alpha_const'] * group['rbar_sum']/group['rbar'] ) 
        group['sum_alpha_r'] += group['alpha'] * group['rbar']
        group['sum_alpha'] += group['alpha']

    def _update_group_norm(self, group, init):
        if self._first_step:
            group['r_x'] = group['r_y'] = group['r_w'] = torch.zeros(1)[0]
        
        group['x_norm'] = torch.stack([torch.norm(p.detach()) for p in group['params']]).norm()
        group['w_norm'] = torch.stack([torch.norm(p.detach()) for p in group['w']]).norm()

        if not (group['decay_factor'] is None):
            group['x_decay_factor'] = min( group['decay_factor'] * group['r_x'] / group['x_norm'], 1.0)
            group['w_decay_factor'] = min( group['decay_factor'] * group['r_w']  / group['w_norm'], 1.0)

    def _update_group_state(self, group, init):
        # treat all layers as one long vector
        if self._first_step:
            group['rbar'] = group['reps_rel'] * (1 + torch.stack([p.norm() for p in group['params']]).norm())
            group['rbar_sum'] = group['rbar'].clone()

            if self.alpha_ver in [2,3,4,5,6]:
                group['alpha'] = group['rbar'] / group['rbar'] # =1
            elif self.alpha_ver in [7]:
                group['alpha'] = group['rbar'] / group['rbar'] # =1
                group['alpha'] = group['alpha'] / (1 - group['momentum'])
            else:
                group['alpha'] = torch.maximum(torch.ones(1)[0],  group['alpha_const'] * group['rbar_sum']/group['rbar'] )
            group['sum_alpha_r'] = group['alpha'] * group['rbar']
            group['sum_alpha'] = group['alpha'].clone()

            group['G_y'] = group['alpha'] * 0 + group['eps']
            group['G_w'] = group['G_y'].clone()

            group['rbar_y'] = group['rbar']
            group['rbar_w'] = group['rbar']
            group['rbar_x'] = group['rbar']
        else:
            #assert (self.opt_ver in [1])
            #curr_d = torch.stack([torch.norm(p.detach() - pi) for p, pi in zip(group['params'], init)]).norm()
            #group['rbar'] = torch.maximum(group['rbar'], curr_d)
            #group['rbar'] = torch.maximum(group['rbar'], group['rbar_y'])
            #group['rbar'] = torch.maximum(group['rbar'], group['rbar_w'])
            #group['rbar'] = torch.maximum(group['rbar'], group['rbar_x'])
            pass

        if self.step_size_ver in [3]:
            group['G_y'] += torch.stack([( group['sum_alpha'] * (p.grad.detach() ** 2) ).sum() for p in group['params']]).sum()
        else:
            group['G_y'] += torch.stack([( (group['alpha'] * p.grad.detach()) ** 2 ).sum() for p in group['params']]).sum()

        if self.step_size_ver in [2,3]:
            group['G_w'] += torch.stack([( group['sum_alpha'] * (p.grad.detach() ** 2) ).sum() for p in group['params']]).sum()
        else:
            group['G_w'] += torch.stack([( (group['alpha'] * p.grad.detach()) ** 2 ).sum() for p in group['params']]).sum()

        assert group['G_y'] > 0 and group['G_w'] > 0, \
            f'DoG cannot work when G is not strictly positive. got: {group["G_y"]}, and: {group["G_w"]}, (first_step = {self._first_step})'
        group['eta_y'] = [group['lr'] * group['rbar'] / torch.sqrt(group['G_y'])] * len(group['params'])
        group['eta_w'] = [group['lr'] * group['rbar'] / torch.sqrt(group['G_w'])] * len(group['params'])

    def _override_init_eta_if_needed(self, group):
        # Override init_eta if needed
        if self._first_step and group['init_eta'] is not None:
            init_eta = group['init_eta']
            logger.info(f'Explicitly setting init_eta value to {init_eta}')
            group['eta'] = [eta * 0 + init_eta for eta in group['eta']]


# class LDoG(DoG):
#     """
#         Layer-wise DoG, as described in:
#        `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule` (Ivgi et al., 2023).
#         LDoG applies the DoG formula defined in the DoG class, but for each layer separately.
#     """
#     def _update_group_state(self, group, init):
#         # treat each layer in the group as a separate block
#         if self._first_step:
#             group['rbar'] = group['reps_rel'] * (1 + torch.stack([p.norm() for p in group['params']]))
#             group['G'] = torch.stack([(p.grad ** 2).sum() for p in group['params']]) + group['eps']
#         else:
#             curr_d = torch.stack([torch.norm(p - pi) for p, pi in zip(group['params'], init)])
#             group['rbar'] = torch.maximum(group['rbar'], curr_d)
#             group['G'] += torch.stack([(p.grad ** 2).sum() for p in group['params']])
#         assert torch.all(group['G'] > 0).item(), \
#             f'DoG cannot work when g2 is not strictly positive. got: {group["G"]}'
#         group['eta'] = list(group['lr'] * group['rbar'] / torch.sqrt(group['G']))
