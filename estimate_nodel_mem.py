from open_lm.model import create_model
import os
import argparse
import torch
models = [
    'layers=8_hidden-dim=288',
    'layers=10_hidden-dim=384',
    'layers=14_hidden-dim=576',
    'layers=18_hidden-dim=704',
    'layers=26_hidden-dim=1312',
    'layers=33_hidden-dim=1792',
    ]

for model in models:
    args = {'model': model, 'model_norm': 'default_layer_norm', 'qk_norm': True, 'rotary_old': False}
    # change args so you can do args.model and args.model_norm
    args = argparse.Namespace(**args)
    model_obj = create_model(args)
    torch.save(model_obj, 'tmp_model')
    # make sure to use amp_bfloat16
    # torch.save(model_obj, 'tmp_model',  _use_new_zipfile_serialization=False)
    size = os.path.getsize('tmp_model')
    os.remove('tmp_model')
    print(f'{model}: {size / 1024 / 1024:.2f} MB')

