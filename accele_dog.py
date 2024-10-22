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

    @torch.no_grad()
    def __init__(self, params, reps_rel: float = 1e-6, lr: float = 1.0, alpha_const: float = 0.5,
                 weight_decay: float = 0.0, decay_to_init: bool = False, decay_to_norm: bool = False, eps: float = 1e-8, init_eta: Optional[float] = None,
                 granularity='all', opt_ver: int = 1, alpha_ver: int = 1, step_size_ver: int = 1, momentum: float = None, decouple_decay: bool = False, decay_after: bool = False, decay_factor: float = None,
                 max_rbar: bool = False, normalize_rbar: bool = False, always_project: bool = False, always_decay: bool = False):
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

        assert not (decay_after and decouple_decay)
        assert not (decay_to_init and decay_to_norm)

        self._first_step = True
        self.opt_ver = opt_ver
        self.alpha_ver = alpha_ver
        self.step_size_ver = step_size_ver
        self.granularity = granularity
        self.save_y = False

        if opt_ver in [2,3]:
            alpha_const = 1.

        defaults = dict(reps_rel=reps_rel, lr=lr, alpha_const=alpha_const, weight_decay=weight_decay, decay_to_init=decay_to_init, decay_to_norm=decay_to_norm, eps=eps, init_eta=init_eta,
                            momentum=momentum, decouple_decay=decouple_decay, decay_after=decay_after, decay_factor=decay_factor, x_decay_factor=[1.0]*len(params), w_decay_factor=[1.0]*len(params), max_rbar=max_rbar, normalize_rbar=normalize_rbar, gm=1.0, always_project=always_project, always_decay=always_decay)
        super(AcceleDoG, self).__init__(params, defaults)

        for group in self.param_groups:
            self._update_dim(group)
            self._init_weight_norm(group)

    def reset(self):
        self._first_step = True

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
            decay_after = group['decay_after']
            decay_to_norm = group['decay_to_norm']

            if first_step:
                for i in range(len(group['params'])):
                    if group['w'][i].device != group['params'][i].device:
                        group['w'][i] = group['w'][i].to(group['params'][i].device)

                for i, p in enumerate(group['params']):
                    if hasattr(p, 'to_normalize') and p.to_normalize:
                        p /= torch.norm_except_dim(p, 2, group['normalize_dim'][i])

                init = group['init_buffer'] = [p.clone().detach_() for p in group['params']]
                #group['w'] = [p.clone().detach_() for p in group['params']]
                for w, p in zip(group['w'], group['params']):
                    w.id_number = p.id_number
                    w.copy_(p)

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

            if weight_decay > 0 and not (decouple_decay or decay_after):
                for i, (p, pi) in enumerate(zip(group['params'], init)):
                    if p.grad is None or (hasattr(p, 'to_normalize') and p.to_normalize) or (hasattr(p, 'skip_decay') and p.skip_decay):
                        continue
                    if (not decay_to_init) and hasattr(p, 'constant_norm') and p.constant_norm:
                        continue
                    if decay_to_init:
                        if not first_step:
                            p.grad.add_(p - pi, alpha=weight_decay)
                    elif decay_to_norm:
                        x_update, _ = self._get_decay_to_norm_update(group, i)

                        pv = p
                        pgv = p.grad
                        if hasattr(p, 'num_of_heads'):
                            pv = p.view(p.num_of_heads, -1)
                            pgv = p.grad.view(p.num_of_heads, -1)
                        pgv.add_(pv * x_update, alpha=-weight_decay)
                    else:
                        p.grad.add_(p, alpha=weight_decay)

            self._update_group_state(group, init)
            self._override_init_eta_if_needed(group)


        for group in self.param_groups:
            init = group['init_buffer']
            momentum = group['momentum']
            decouple_decay = group['decouple_decay']
            decay_after = group['decay_after']

            if weight_decay > 0 and decouple_decay:
                self._do_decouple_decay(group)

            for i, (x, w, eta_y, eta_w, alpha) in enumerate(zip(group['params'], group['w'], group['eta_y'], group['eta_w'], group['alpha'])):
                if x.grad is None:
                    continue
                else:
                    wv = w
                    xv = x
                    x_grad = x.grad.detach()
                    if hasattr(x, 'num_of_heads'):
                        wv = w.view(x.num_of_heads, -1)
                        xv = x.view(x.num_of_heads, -1)
                        x_grad = x_grad.view(x.num_of_heads, -1)

                    wv.mul_(momentum)
                    if wv.device != xv.device:
                        logging.info(f"wv.device={wv.device}, xv.device={xv.device}")
                    wv.add_(xv, alpha=1-momentum)
                    wv.add_(x_grad * eta_w * alpha, alpha=-1 )

                    xv.add_(x_grad * eta_y, alpha=-1)

                    if hasattr(x, 'to_normalize') and x.to_normalize:
                        w /= torch.norm_except_dim(w, 2, group['normalize_dim'][i])

            self._normalize_weight(group)

            if weight_decay > 0 and decay_after:
                self._do_decouple_decay(group)

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
                    wv = w
                    xv = x
                    if hasattr(x, 'num_of_heads'):
                        wv = w.view(x.num_of_heads, -1)
                        xv = x.view(x.num_of_heads, -1)

                    xv.mul_( -tau + 1)
                    xv.add_(wv * tau)

                    if hasattr(x, 'to_normalize') and x.to_normalize:
                        x /= torch.norm_except_dim(x, 2, group['normalize_dim'][i])

            self._normalize_weight(group)
            self._update_group_dist(group, 'x', group['params'])

        self._first_step = False

        return loss

    def _do_decouple_decay(self, group):
        weight_decay = group['weight_decay']
        decay_to_init = group['decay_to_init']
        decay_to_norm = group['decay_to_norm']

        for i, (x, w, pi, x_decay_factor, w_decay_factor) in enumerate(zip(group['params'], group['w'], group['init_buffer'], group['x_decay_factor'], group['w_decay_factor'])):
            if x.grad is None:
                continue
            if hasattr(x, 'to_normalize') and x.to_normalize:
                continue
            if hasattr(x, 'skip_decay') and x.skip_decay:
                continue
            if (not decay_to_init) and hasattr(x, 'constant_norm') and x.constant_norm:
                continue

            wv = w
            xv = x
            if hasattr(x, 'num_of_heads'):
                wv = w.view(x.num_of_heads, -1)
                xv = x.view(x.num_of_heads, -1)

            if decay_to_init:
                if not self._first_step:
                    x.add_(x - pi, alpha=-weight_decay)
                    w.add_(w - pi, alpha=-weight_decay)
            elif decay_to_norm:
                x_update, w_update = self._get_decay_to_norm_update(group, i)

                xv.add_(xv * x_update, alpha=weight_decay )
                wv.add_(wv * w_update, alpha=weight_decay )
            else:
                xv.add_(xv * x_decay_factor, alpha=-weight_decay )
                wv.add_(wv * w_decay_factor, alpha=-weight_decay )

    def _get_decay_to_norm_update(self, group, i):
        always_decay = group['always_decay']

        x_norm = self._norm(group['params'][i], dim=group['dim'][i])
        x_norm = torch.maximum(x_norm, torch.ones_like(x_norm) * group['eps'])

        w_norm = self._norm(group['w'][i], dim=group['dim'][i])
        w_norm = torch.maximum(w_norm, torch.ones_like(w_norm) * group['eps'])

        x_update = group['init_x_norm'][i] / x_norm - 1
        w_update = group['init_w_norm'][i] / w_norm - 1

        if not always_decay:
            x_update.where(x_update <= 0., 0.)
            w_update.where(w_update <= 0., 0.)

        return x_update, w_update

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
        assert self._first_step

        group['dim'] = [None] * len(group['params'])
        group['norm_dim'] = [None] * len(group['params'])
        group['normalize_dim'] = [-1] * len(group['params'])

        for i, p  in enumerate(group['params']):
            if hasattr(p, 'to_normalize') and p.to_normalize:
                if p.to_normalize in ['first', 'output']:
                    group['normalize_dim'][i] = 0
                elif p.to_normalize in ['last', 'input']:
                    group['normalize_dim'][i] = p.dim()-1

            if self.granularity == 'param':
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

            group['x_norm'][i] = torch.maximum(group['x_norm'][i], torch.ones_like(group['x_norm'][i]) * group['eps'])
            group['w_norm'][i] = torch.maximum(group['w_norm'][i], torch.ones_like(group['w_norm'][i]) * group['eps'])

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
                    group['G_y'][i] = self._norm(group['params'][i], dim=group['dim'][i]) * 0
                else:
                    group['G_y'][i] = group['alpha'][i] * 0
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
                group['G_y'][0] += torch.stack([( group['sum_alpha'][0] * ((p.grad.detach() * group['gm']) ** 2) ).sum() for p in group['params']]).sum()
            else:
                group['G_y'][0] += torch.stack([( (group['alpha'][0] * p.grad.detach() * group['gm']) ** 2 ).sum() for p in group['params']]).sum()

            if self.step_size_ver in [2,3]:
                group['G_w'][0] += torch.stack([( group['sum_alpha'][0] * ((p.grad.detach() * group['gm']) ** 2) ).sum() for p in group['params']]).sum()
            else:
                group['G_w'][0] += torch.stack([( (group['alpha'][0] * p.grad.detach() * group['gm']) ** 2 ).sum() for p in group['params']]).sum()

            for i in range(1, len(group['params'])):
                group['G_y'][i] = group['G_y'][0]
                group['G_w'][i] = group['G_w'][0]
        elif self.granularity == 'param':
            for i in range(len(group['params'])):
                if self.step_size_ver in [3]:
                    group['G_y'][i] += group['sum_alpha'][i] * (self._sum( ((group['params'][i].grad.detach() * group['gm']) ** 2), dim=group['dim'][i]))
                else:
                    if self._first_step:
                        logging.info( "shapes: " + str((group['G_y'][i].shape, group['alpha'][i].shape, group['params'][i].grad.shape)) )
                    group['G_y'][i] += (group['alpha'][i]**2) * (self._sum( (group['params'][i].grad.detach() * group['gm']) ** 2, dim=group['dim'][i]))

                if self.step_size_ver in [2,3]:
                    group['G_w'][i] += group['sum_alpha'][i] * (self._sum( ((group['params'][i].grad.detach() * group['gm']) ** 2), dim=group['dim'][i]))
                else:
                    group['G_w'][i] += (group['alpha'][i]**2) * (self._sum( (group['params'][i].grad.detach() * group['gm']) ** 2, dim=group['dim'][i]))

        for i in range(len(group['params'])): 
            assert (group['G_y'][i] >= 0).all() and (group['G_w'][i] >= 0).all(), \
                f'DoG cannot work when G is not strictly positive. got: {group["G_y"][i]}, and: {group["G_w"][i]}, (first_step = {self._first_step})'

        if self.granularity == 'all':
            group['eta_y'] = [group['lr'] * group['rbar'][0] / (torch.sqrt(group['G_y'][0]) + group['eps'])] * len(group['params'])
            group['eta_w'] = [group['lr'] * group['rbar'][0] / (torch.sqrt(group['G_w'][0]) + group['eps'])] * len(group['params'])
        elif self.granularity == 'param':
            group['eta_y'] = [group['lr'] * group['rbar'][i] / (torch.sqrt(group['G_y'][i]) + group['eps']) for i in range(len(group['params']))]
            group['eta_w'] = [group['lr'] * group['rbar'][i] / (torch.sqrt(group['G_w'][i]) + group['eps']) for i in range(len(group['params']))]

    def _override_init_eta_if_needed(self, group):
        # Override init_eta if needed
        if self._first_step and group['init_eta'] is not None:
            init_eta = group['init_eta']
            logger.info(f'Explicitly setting init_eta value to {init_eta}')
            group['eta_y'] = [eta * 0 + init_eta for eta in group['eta_y']]
            group['eta_w'] = [eta * 0 + init_eta for eta in group['eta_w']]

    def _init_weight_norm(self, group):
        group['w'] = [p.clone().detach_() for p in group['params']]
        for i in range(len(group['w'])):
            w, x = group['w'][i], group['params'][i]
            logging.info(f"{i}: w.device={w.device}, x.device={x.device}")

        group['init_x_norm'] = [ None for p in group['params']]
        group['init_w_norm'] = [ None for p in group['params']]
        for i, (x, w) in enumerate(zip(group['params'], group['w'])):
            #if not (hasattr(x, 'constant_norm') and x.constant_norm):
            #    continue

            group['init_x_norm'][i] = self._norm(x,dim=group['norm_dim'][i]).mean()
            group['init_w_norm'][i] = group['init_x_norm'][i]

            logging.info( "norm: " + str(group['init_x_norm'][i]) )

        self._normalize_weight(group)

    def _normalize_weight(self, group):
        for i, (x, w) in enumerate(zip(group['params'], group['w'])):
            if not (hasattr(x, 'constant_norm') and x.constant_norm):
                continue

            wv = w
            xv = x
            if hasattr(x, 'num_of_heads'):
                wv = w.view(x.num_of_heads, -1)
                xv = x.view(x.num_of_heads, -1)

            x_norm = self._norm(x,dim=group['norm_dim'][i])
            w_norm = self._norm(w,dim=group['norm_dim'][i])

            x_factor = group['init_x_norm'][i] / x_norm
            w_factor = group['init_w_norm'][i] / w_norm

            if not group['always_project']:
                x_factor = x_factor.where(x_factor <= 1., 1.)
                w_factor = w_factor.where(w_factor <= 1., 1.)

            xv.mul_( x_factor )
            wv.mul_( w_factor )

            if self._first_step:
                logging.info( "normalize factors: " + str((x_factor, w_factor)) )

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