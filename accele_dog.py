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
                 granularity='all', opt_ver: int = 1, alpha_ver: int = 1, step_size_ver: int = 1, momentum: float = None, decouple_decay: bool = False, decay_factor: float = None,
                 max_rbar: bool = False, normalize_rbar: bool = False):
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

        if momentum is None:
            momentum = 1.0

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
                            momentum=momentum, decouple_decay=decouple_decay, decay_factor=decay_factor, x_decay_factor=[1.0]*len(params), w_decay_factor=[1.0]*len(params), max_rbar=max_rbar, normalize_rbar=normalize_rbar)
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
                rbar.append((group['rbar'][0].norm()).item())
                eta.append((group['eta_y'][0].sum()/group['eta_y'][0].numel()).item())
                alpha.append((group['alpha'][0].sum()/group['alpha'][0].numel()).item())
                sum_alpha_r.append((group['sum_alpha_r'][0].sum()/group['sum_alpha_r'][0].numel()).item())

            return dict(rbar = np.array(rbar),
                eta = np.array(eta),
                alpha = np.array(alpha),
                sum_alpha_r=np.array(sum_alpha_r))
        else:
            for group in self.param_groups:
                return dict(rbar = group['rbar'][0].item(),
                    rbar_x = group['rbar_x'][0].item(),
                    rbar_y = group['rbar_y'][0].item(),
                    rbar_w = group['rbar_w'][0].item(),
                    eta = group['eta_y'][0].item(),
                    #eta_w = group['eta_w'][0].item(),
                    alpha = group['alpha'][0].item(),
                    sum_alpha_r=group['sum_alpha_r'][0].item())

    # @torch.no_grad()
    # def get_average_weight(self):
    #     y_bar = {}
    #     for group in self.param_groups:
    #         for p in group['y_bar']:
    #             y_bar[p.id_number] = p.clone().detach_()
    #     return y_bar

    # @torch.no_grad()
    # def get_y(self):
    #     y = {}
    #     for group in self.param_groups:
    #         for p in group['y']:
    #             y[p.id_number] = p.clone().detach_()
    #     return y

    # @torch.no_grad()
    # def get_w(self):
    #     w = {}
    #     for group in self.param_groups:
    #         for p in group['w']:
    #             w[p.id_number] = p.clone().detach_()
    #     return w

    # @torch.no_grad()
    # def get_x(self):
    #     x = {}
    #     for group in self.param_groups:
    #         for p in group['params']:
    #             x[p.id_number] = p.clone().detach_()
    #     return x

    # @torch.no_grad()
    # def get_gamma(self, idx=0):
    #     idx = idx % len(self.param_groups)
    #     for i, group in enumerate(self.param_groups):
    #         if not i == idx:
    #             continue
    #         alpha_r, sum_alpha_r = group['alpha'] * group['rbar'], group['sum_alpha_r']
    #         return alpha_r / sum_alpha_r

    #     assert False

    # @torch.no_grad()
    # def average_step(self, group):
    #     if self.save_y:
    #         for y, p in zip(group['y'], group['params']):
    #             y.copy_( p )

    #     alpha_r, sum_alpha_r = group['alpha'] * group['rbar'], group['sum_alpha_r']
    #     for y_bar, y in zip(group['y_bar'], group['params']):
    #         y_bar.mul_( 1 - (alpha_r / sum_alpha_r) )
    #         y_bar.add_( (alpha_r / sum_alpha_r) * y )

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
                self._update_dim(group)

                for i, p in enumerate(group['params']):
                    if hasattr(p, 'to_normalize') and p.to_normalize:
                        p /= torch.norm_except_dim(p, 2, group['normalize_dim'][i])

                init = group['init_buffer'] = [p.clone().detach_() for p in group['params']]
                group['w'] = [p.clone().detach_() for p in group['params']]
                for w, p in zip(group['w'], group['params']):
                    w.id_number = p.id_number

                # group['y_bar'] = [p.clone().detach_().zero_() for p in group['params']]
                # for y, p in zip(group['y_bar'], group['params']):
                #     y.id_number = p.id_number

                # if self.save_y:
                #     group['y'] = [p.clone().detach_().zero_() for p in group['params']]
                #     for y, p in zip(group['y'], group['params']):
                #         y.id_number = p.id_number
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

            weight_decay = group['weight_decay']
            decouple_decay = group['decouple_decay']
            decay_to_init = group['decay_to_init']
            momentum = group['momentum']

            for x, w, pi, x_decay_factor, w_decay_factor in zip(group['params'], group['w'], init, group['x_decay_factor'], group['w_decay_factor']):
                if x.grad is None:
                    continue
                else:
                    if weight_decay > 0 and decouple_decay and not (hasattr(x, 'to_normalize') and x.to_normalize):
                        if decay_to_init:
                            if not first_step:
                                x.add_(x - pi, alpha=-weight_decay)
                                w.add_(w - pi, alpha=-weight_decay)
                        else:
                            x.add_(x * x_decay_factor, alpha=-weight_decay )
                            w.add_(w * w_decay_factor, alpha=-weight_decay )

            for i, (x, w, eta_y, eta_w, alpha) in enumerate(zip(group['params'], group['w'], group['eta_y'], group['eta_w'], group['alpha'])):
                if x.grad is None:
                    continue
                else:
                    w.mul_(momentum)
                    w.add_(x, alpha=1-momentum)
                    w.add_(x.grad.detach() * eta_w * alpha, alpha=-1 )

                    x.add_(x.grad.detach() * eta_y, alpha=-1)

                    if hasattr(x, 'to_normalize') and x.to_normalize:
                        w /= torch.norm_except_dim(w, 2, group['normalize_dim'][i])

            # self.average_step( group )

            self._update_group_dist(group, 'y', group['params'])
            self._update_group_dist(group, 'w', group['w'])

            #recalculate tau
            self._update_group_total_dist(group, init)

            taus = [0.0] * len(group['params'])
            for i in range(len(group['params'])):
                if self.opt_ver == 2:
                    taus[i] = group['alpha'][i] * group['rbar'][i] / group['sum_alpha_r'][i]
                if self.opt_ver == 3:
                    taus[i] = group['alpha'][i] / group['sum_alpha'][i]
                else:
                    taus[i] = 1 / group['alpha'][i]
                if self.granularity == 'all':
                    break
            
            if self.granularity == 'all':
                for i in range(len(group['params'])):
                    taus[i] = taus[0]

            for i, (x, w, eta_y, eta_w, tau) in enumerate(zip(group['params'], group['w'], group['eta_y'], group['eta_w'], taus)):
                if x.grad is None:
                    continue
                else:
                    x.mul_( -tau + 1)
                    x.add_(w * tau)

                    if hasattr(x, 'to_normalize') and x.to_normalize:
                        x /= torch.norm_except_dim(x, 2, group['normalize_dim'][i])

            self._update_group_dist(group, 'x', group['params'])

        self._first_step = False

        return loss

    def _update_group_dist(self, group, name, param):
        for i in range(len(group['params'])):
            if self.granularity == 'all':
                curr_d = torch.stack([(p.detach() - pi).norm() for p, pi in zip(param, group['init_buffer'])]).norm()
            elif self.granularity == 'param':
                p, pi = param[i], group['init_buffer'][i]
                curr_d = self._norm(p.detach() - pi, dim=group['dim'][i])
                if group['max_rbar']:
                    curr_d = curr_d.max()
                elif group['normalize_rbar']:
                    curr_d = torch.norm(curr_d) / (curr_d.numel() ** 0.5)
            group['rbar_' + name][i] = torch.maximum(group['rbar_' + name][i], curr_d)
            group['r_' + name][i] = curr_d
            if self.granularity == 'all':
                break

        if self.granularity == 'all': 
            for i in range(1, len(group['params'])):
                group['rbar_' + name][i] = group['rbar_' + name][0]
                group['r_' + name][i] = curr_d

    def _update_group_total_dist(self, group, init):
        for i in range(len(group['params'])):
            prev_rbar = group['rbar'][i]
            group['rbar'][i] = torch.maximum(group['rbar'][i], group['rbar_w'][i])
            group['rbar_sum'][i] += group['rbar'][i]

            if self.alpha_ver in [2]:
                group['alpha'][i] += 1
            elif self.alpha_ver in [3]:
                group['alpha'][i] = (1 / group['momentum']) ** torch.maximum(torch.ones(1)[0], group['alpha_const'] * group['rbar_sum'][i]/group['rbar'][i] )
            elif self.alpha_ver in [4]:
                group['alpha'][i] *= prev_rbar / (group['rbar'][i] * group['momentum'])
            elif self.alpha_ver in [5]:
                group['alpha'][i] /= group['momentum']
            elif self.alpha_ver in [6]:
                group['alpha'][i] = ((1-group['momentum']) / group['momentum']) * group['sum_alpha_r'][i] / group['rbar'][i]
            elif self.alpha_ver in [7]:
                pass
            else:
                group['alpha'][i] = torch.maximum(torch.ones(1)[0], group['alpha_const'] * group['rbar_sum'][i]/group['rbar'][i] ) 
            group['sum_alpha_r'][i] += group['alpha'][i] * group['rbar'][i]
            group['sum_alpha'][i] += group['alpha'][i]
            if self.granularity == 'all':
                break

        if self.granularity == 'all':
            for i in range(1, len(group['params'])):
                group['rbar'][i] = group['rbar'][0]
                group['rbar_sum'][i] = group['rbar_sum'][0]
                group['alpha'][i] = group['alpha'][0]
                group['sum_alpha_r'][i] = group['sum_alpha_r'][0]
                group['sum_alpha'][i] = group['sum_alpha'][0]

    def _update_dim(self, group):
        if not self._first_step:
            return

        group['dim'] = [None] * len(group['params'])
        group['normalize_dim'] = [-1] * len(group['params'])

        for i, p  in enumerate(group['params']):
            if hasattr(p, 'to_normalize') and p.to_normalize:
                if p.to_normalize in ['first', 'output']:
                    group['normalize_dim'][i] = 0
                elif p.to_normalize in ['last', 'input']:
                    group['normalize_dim'][i] = p.dim()-1

            if self.granularity == 'param':
                norm_dim = [j for j in range(p.dim())]
                if hasattr(p, 'per_inout'):
                    if p.per_inout == 'first' or p.per_inout == 'output':
                        norm_dim.remove(0)
                    elif p.per_inout == 'last' or p.per_inout == 'input':
                        norm_dim.remove(p.dim()-1)
                    elif p.per_inout == 'all':
                        norm_dim = []
                group['dim'][i] = tuple(norm_dim)
        

    def _update_group_norm(self, group, init):
        if self._first_step:
            group['r_x'] = group['r_y'] = group['r_w'] = [torch.zeros(1)[0]] * len(group['params'])
            group['x_decay_factor'] = [0.0] * len(group['params'])
            group['w_decay_factor'] = [0.0] * len(group['params'])
            group['x_norm'] = [0.0] * len(group['params'])
            group['w_norm'] = [0.0] * len(group['params'])
        
        for i in range(len(group['params'])):
            if self.granularity == 'all':
                group['x_norm'][i] = torch.stack([torch.norm(p.detach()) for p in group['params']]).norm()
                group['w_norm'][i] = torch.stack([torch.norm(p.detach()) for p in group['w']]).norm()
            elif self.granularity == 'param':
                group['x_norm'][i] = self._norm(group['params'][i], dim=group['dim'][i])
                group['w_norm'][i] = self._norm(group['w'][i], dim=group['dim'][i])

            if not (group['decay_factor'] is None):
                group['x_decay_factor'][i] = torch.minimum( group['decay_factor'] * group['r_x'][i] / group['x_norm'][i], torch.ones_like(group['x_norm'][i]))
                group['w_decay_factor'][i] = torch.minimum( group['decay_factor'] * group['r_w'][i]  / group['w_norm'][i], torch.ones_like(group['w_norm'][i]))
            if self.granularity == 'all':
                break
        
        if self.granularity == 'all':
            for i in range(1, len(group['params'])):
                group['x_norm'][i] = group['x_norm'][0]
                group['w_norm'][i] = group['w_norm'][0]
                group['x_decay_factor'][i] = group['x_decay_factor'][0]
                group['w_decay_factor'][i] = group['w_decay_factor'][0]

    def _update_group_state(self, group, init):
        # treat all layers as one long vector
        if self._first_step:
            if self.granularity == 'all':
                group['rbar'] = [group['reps_rel'] * (1 + torch.stack([p.norm() for p in group['params']]).norm())] * len(group['params'])
            elif self.granularity == 'param':
                if (not group['max_rbar']) and (not group['normalize_rbar']):
                    group['rbar'] = [group['reps_rel'] * (1 + self._norm(p, dim=group['dim'][i])) for i,p in enumerate(group['params'])]
                else:
                    group['rbar'] = [group['reps_rel'] * (1 + self._norm(p,dim=group['dim'][i])).max() for i,p in enumerate(group['params'])]
            
            group['rbar_sum'] = [0.] * len(group['params'])
            group['alpha'] = [0.] * len(group['params'])
            group['sum_alpha_r'] = [0.] * len(group['params'])
            group['sum_alpha'] = [0.] * len(group['params'])
            group['G_y'] = [0.] * len(group['params'])
            group['G_w'] = [0.] * len(group['params'])
            group['rbar_y'] = [0.] * len(group['params'])
            group['rbar_w'] = [0.] * len(group['params'])
            group['rbar_x'] = [0.] * len(group['params'])

            for i in range(len(group['params'])):
                group['rbar_sum'][i] = group['rbar'][i].clone()

                if self.alpha_ver in [2,3,4,5,6]:
                    group['alpha'][i] = group['rbar'][i] / group['rbar'][i] # =1
                elif self.alpha_ver in [7]:
                    group['alpha'][i] = group['rbar'][i] / group['rbar'][i] # =1
                    group['alpha'][i] = group['alpha'][i] / (1 - group['momentum'])
                else:
                    group['alpha'][i] = torch.maximum(torch.ones(1)[0],  group['alpha_const'] * group['rbar_sum'][i]/group['rbar'][i] )
                group['sum_alpha_r'][i] = group['alpha'][i] * group['rbar'][i]
                group['sum_alpha'][i] = group['alpha'][i].clone()

                if self.granularity == 'param' and (group['max_rbar'] or group['normalize_rbar']):
                    group['G_y'][i] = self._norm(group['params'][i], dim=group['dim'][i]) * 0 + group['eps']
                else:
                    group['G_y'][i] = group['alpha'][i] * 0 + group['eps']
                group['G_w'][i] = group['G_y'][i].clone()

                group['rbar_y'][i] = group['rbar'][i]
                group['rbar_w'][i] = group['rbar'][i]
                group['rbar_x'][i] = group['rbar'][i]
                
                if self.granularity == 'all':
                    break

            if self.granularity == 'all':
                for i in range(1, len(group['params'])):
                    group['rbar_sum'][i] = group['rbar_sum'][i]
                    group['alpha'][i] = group['alpha'][i]
                    group['sum_alpha_r'][i] = group['sum_alpha_r'][i]
                    group['sum_alpha'][i] = group['sum_alpha'][i]
                    group['G_y'][i] = group['G_y'][i]
                    group['G_w'][i] = group['G_w'][i]
                    group['rbar_y'][i] = group['rbar_y'][i]
                    group['rbar_w'][i] = group['rbar_w'][i]
                    group['rbar_x'][i] = group['rbar_x'][i]
        else:
            #assert (self.opt_ver in [1])
            #curr_d = torch.stack([torch.norm(p.detach() - pi) for p, pi in zip(group['params'], init)]).norm()
            #group['rbar'] = torch.maximum(group['rbar'], curr_d)
            #group['rbar'] = torch.maximum(group['rbar'], group['rbar_y'])
            #group['rbar'] = torch.maximum(group['rbar'], group['rbar_w'])
            #group['rbar'] = torch.maximum(group['rbar'], group['rbar_x'])
            pass

        if self.granularity == 'all':
            if self.step_size_ver in [3]:
                group['G_y'][0] += torch.stack([( group['sum_alpha'][0] * (p.grad.detach() ** 2) ).sum() for p in group['params']]).sum()
            else:
                group['G_y'][0] += torch.stack([( (group['alpha'][0] * p.grad.detach()) ** 2 ).sum() for p in group['params']]).sum()

            if self.step_size_ver in [2,3]:
                group['G_w'][0] += torch.stack([( group['sum_alpha'][0] * (p.grad.detach() ** 2) ).sum() for p in group['params']]).sum()
            else:
                group['G_w'][0] += torch.stack([( (group['alpha'][0] * p.grad.detach()) ** 2 ).sum() for p in group['params']]).sum()

            for i in range(1, len(group['params'])):
                group['G_y'][i] = group['G_y'][0]
                group['G_w'][i] = group['G_w'][0]
        elif self.granularity == 'param':
            for i in range(len(group['params'])):
                if self.step_size_ver in [3]:
                    group['G_y'][i] += group['sum_alpha'][i] * (self._sum( (group['params'][i].grad.detach() ** 2), dim=group['dim'][i]))
                else:
                    if self._first_step:
                        logging.info( "shapes: " + str((group['G_y'][i].shape, group['alpha'][i].shape, group['params'][i].grad.shape)) )
                    group['G_y'][i] += (group['alpha'][i]**2) * (self._sum( (group['params'][i].grad.detach()) ** 2, dim=group['dim'][i]))

                if self.step_size_ver in [2,3]:
                    group['G_w'][i] += group['sum_alpha'][i] * (self._sum( (group['params'][i].grad.detach() ** 2), dim=group['dim'][i]))
                else:
                    group['G_w'][i] += (group['alpha'][i]**2) * (self._sum( (group['params'][i].grad.detach()) ** 2, dim=group['dim'][i]))

        for i in range(len(group['params'])): 
            assert (group['G_y'][i] > 0).all() and (group['G_w'][i] > 0).all(), \
                f'DoG cannot work when G is not strictly positive. got: {group["G_y"][i]}, and: {group["G_w"][i]}, (first_step = {self._first_step})'

        if self.granularity == 'all':
            group['eta_y'] = [group['lr'] * group['rbar'][0] / torch.sqrt(group['G_y'][0])] * len(group['params'])
            group['eta_w'] = [group['lr'] * group['rbar'][0] / torch.sqrt(group['G_w'][0])] * len(group['params'])
        elif self.granularity == 'param':
            group['eta_y'] = [group['lr'] * group['rbar'][i] / torch.sqrt(group['G_y'][i]) for i in range(len(group['params']))]
            group['eta_w'] = [group['lr'] * group['rbar'][i] / torch.sqrt(group['G_w'][i]) for i in range(len(group['params']))]

    def _override_init_eta_if_needed(self, group):
        # Override init_eta if needed
        if self._first_step and group['init_eta'] is not None:
            init_eta = group['init_eta']
            logger.info(f'Explicitly setting init_eta value to {init_eta}')
            group['eta_y'] = [eta * 0 + init_eta for eta in group['eta_y']]
            group['eta_w'] = [eta * 0 + init_eta for eta in group['eta_w']]

    def _sum(self, p, dim):
        if dim == tuple():
            return p
        return p.sum(dim=dim, keepdim=True)

    def _norm(self, p, dim):
        if dim == tuple():
            return p
        return p.norm(dim=dim, keepdim=True)