"""
Based on from experiment starter code.
"""
import numpy as np
import torch
import logging
from copy import deepcopy

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model



class ModelAverager(object):
    def __init__(self, model, methods: str):
        # self.t = 1
        self.model = model
        self.avgs_dict = {}
        for method in methods.split(','):
            args = method.split('f')
            method_name = args[0][:-1]  if args[0].endswith('_') else args[0]
            freq = int(args[1]) if len(args) > 1 else 1
            # T = total_steps // freq # T is the total numner of steps
            self.avgs_dict[method] = Averager(model, method_name, freq)

    def step(self):
        for avg in self.avgs_dict.values():
            avg.step()


class Averager(object):
    def __init__(self, model, method, freq):
        self.model = model
        self.method = method
        # self.T = T
        self.update_counter = 1
        self.step_counter = 1
        self.freq = freq
        method, *args = method.split('_')
        self.method = method
        if method == 'none':
            self.av_model = model
            return
        else:
            self.av_model = deepcopy(model)
        if method == 'poly':
            self.eta = 0.0 if not args else float(args[0])
            self.freq = 1 if (len(args) < 2 or not args) else int(args[1])
        elif method == 'ema':
            self.gamma = 0.99 if not args else float(args[0])
        # elif method == 'suffix':
        #     self.suffix_steps = self.T -int(self.T/float(args[0])) if args else 0 # alpha is 1/args[0]
            # comment out log/sqrt freq for now
            # if sub_sample == 'log':
            #     self.freq = int((self.T - self.suffix_steps) / np.log(self.T - self.suffix_steps))
            # elif sub_sample == 'sqrt':
            #     self.freq = int(np.sqrt(self.T - self.suffix_steps))
            # elif sub_sample.startswith('freq'):
            #     self.freq = int(sub_sample.split('_')[-1])
            # else:
            #     self.freq = 0
        elif method == 'cosine':
            pass
        # elif method == 'degree':
        #     self.power = float(args[0])
        #     self.start = int((1 - 1/float(args[1]))*self.T) if len(args) > 1 else 0
        else:
            print(f'Unknown averaging method {method}')

    def step(self):
        if self.update_counter != self.freq:
            pass
        else:
            self.update()
        self.update_counter += 1
        if self.update_counter > self.freq:
            self.update_counter = 1
        return

    def update(self):
        method = self.method
        if method == 'none':
            return
        t = self.step_counter
        # model_sd is the current model state dict
        # av_sd is the averaged model state dict
        model_sd = self.model.state_dict()
        av_sd = self.av_model.state_dict()
        if self.method == 'cosine' or self.method == 'degree':
            pass
            # model_place_holder_sd = self.model_place_holder.state_dict()
        for k in model_sd.keys():
            if isinstance(av_sd[k], (torch.LongTensor, torch.cuda.LongTensor)):  
                # these are buffers that store how many batches batch norm has seen so far
                av_sd[k].copy_(model_sd[k])
                continue
            if method == 'poly':
                # the update rule is: new_average = (1 - (eta + 1) / (eta + t)) * old_average + (eta + 1) / (eta + t) * current_model which is eq(10) in https://arxiv.org/pdf/1212.1824.pdf
                av_sd[k].mul_(1 - ((self.eta + 1) / (self.eta + t))).add_(
                    model_sd[k], alpha=(self.eta + 1) / (self.eta + t)
                )
                
            # elif method == 'suffix':
            #     # the update rule is: new_average = average of the last suffix_steps steps
            #     if t > self.suffix_steps:
            #         iterates_to_avg = t-self.suffix_steps
            #         if self.freq > 0:
            #             iterates_to_avg = int(iterates_to_avg / self.freq)
            #         av_sd[k].mul_(
            #             1-1/(iterates_to_avg)
            #             ).add_(
            #             model_sd[k], alpha=1/(iterates_to_avg)
            #             )
            # elif method == 'cosine':
            #     # the weight of the t iteration is 0.5(cos(pi*(t-1)/T) + cos(pi*t/T))))))
            #     av_sd[k].add_(model_sd[k], 
            #     alpha=0.5*(np.cos(np.pi * (t-1) / self.T) -  np.cos(np.pi * t / self.T))
            #     )
            # elif method == 'degree':
            #     # the weitght of the t iteration is ((T - t + 1) / (T - start)) ** (power) - ((T - t) / (T - start)) ** (power) if t >= start else 1
            #     if t >= self.start:
            #         av_sd[k].add_(model_sd[k], 
            #         alpha=((self.T - (t-1)) / (self.T )) ** (self.power) - ((self.T - t) / (self.T)) ** (self.power) 
            #         )
        self.step_counter += 1
        

    def reset(self):
        self.step_counter = 2

    @property
    def averaged_model(self):
        return self.av_model

    def get_state_dict_avg(self):
        state_dict = {'update_counter': self.update_counter, 
        'step_counter': self.step_counter, 
        'freq': self.freq, 
        # 'av_model': self.av_model,
        'av_model_sd': unwrap_model(self.av_model).state_dict(), # unwrap model to get the state dict
        # 'T': self.T if hasattr(self, 'T') else None,
        'method': self.method,
        'eta': self.eta if hasattr(self, 'eta') else None,
        'gamma': self.gamma if hasattr(self, 'gamma') else None,
        'suffix_steps': self.suffix_steps if hasattr(self, 'suffix_steps') else None,
        'power': self.power if hasattr(self, 'power') else None,
        'start': self.start if hasattr(self, 'start') else None,
        } # no need to save model since it is saved in the model itself
        return state_dict
    
    def load_state_dict_avg(self, state_dict):
        self.update_counter = state_dict['update_counter']
        self.step_counter = state_dict['step_counter']
        self.freq = state_dict['freq']
        self.method = state_dict['method']
        self.av_model.load_state_dict(state_dict['av_model_sd'])
        # if hasattr(self, 'T'):
        #     self.T = state_dict['T']
        if hasattr(self, 'eta'):
            self.eta = state_dict['eta']
        if hasattr(self, 'gamma'):
            self.gamma = state_dict['gamma']
        if hasattr(self, 'suffix_steps'):
            self.suffix_steps = state_dict['suffix_steps']
        if hasattr(self, 'power'):
            self.power = state_dict['power']
        if hasattr(self, 'start'):
            self.start = state_dict['start']
        # self.state_dict = state_dict
