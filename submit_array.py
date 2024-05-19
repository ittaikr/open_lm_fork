import yaml
import os
import shutil
import itertools
import argparse
import subprocess
import datetime
import pandas as pd
import numpy as np
from copy import deepcopy
from time import sleep
import typing as tp

import submitit
import pickle

from dataclasses import dataclass, asdict


@dataclass
class Slurm_Config:
    job_name: str = None

    partition: str = None
    setup: tp.List[str] = None

    time: int = 24 * 60
    stderr_to_stdout: bool = True
    
    nodes: int = 1
    ntasks_per_node: int =1
    cpus_per_task: int = 2
    gpus_per_task: int= 0
    mem: str = "8G"

    constraint: str = None
    exclude: str = None


def format_key(k):
    return '_'.join([kk[:3] for kk in k.split('_')])


def format_value(v):
    return str(v).replace(' ', '').replace('/', '-')


def unwind_grid(grid_dict):
    list_keys = [k for k in grid_dict.keys() if k.startswith('LIST')]
    grid = []
    for k in list_keys:
        subgrid_list = grid_dict.pop(k)
        subgrid = []
        for subgrid_dict in subgrid_list:
            subgrid.extend([tuple(zip(*el)) for el in unwind_grid(subgrid_dict)])
        grid.append(tuple(set(subgrid)))
    for k, v in grid_dict.items():
        if isinstance(v, str):
            v = list(eval(v))
        if not isinstance(v, list) or isinstance(v, tuple):
            v = [v]
        grid.append(tuple((k, vv) for vv in v))
    return list(itertools.product(*grid))


def grid_to_str(list_of_dicts):
    nunique = pd.DataFrame(list_of_dicts).nunique(dropna=False)
    nunique = nunique[nunique > 1]
    assert nunique.prod() == len(list_of_dicts)
    return ' x '.join(f'{n} {k}' for k, n in nunique.items())


def expand_tuple_keys(dict_with_tuples):
    tuple_keys = [k for k in dict_with_tuples if isinstance(k, tuple)]
    if len(tuple_keys) == 0:
        return dict_with_tuples
    else:
        for kk in tuple_keys:
            vv = dict_with_tuples.pop(kk)
            dict_with_tuples.update(dict(zip(kk, vv)))
        return expand_tuple_keys(dict_with_tuples)


def call_main_with_args(*args):
    print(type(args), args)
    config, name, logs = args
    args = ['--name', name, '--logs', './'+logs, '--config', config]
    print(logs)
    print(args)

    from open_lm.main import main as main_main
    main_main(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jobfile', type=str,
                        help='path to YAML file containing job configuration')
    parser.add_argument('-s', '--script', type=str, help='name of yaml file that is used to create the script to provide to sbatch commands')
    parser.add_argument('-y', '--yes', action='store_true', help='confirm submission without viewing grid details first')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing results directory')
    parser.add_argument('-d', '--dry_run', action='store_true',
                        help='prints the sbatch commands instead of executing them, and does not create an output folder')
    parser.add_argument('-r', '--rerun', action='store_true',
                        help='re-submits all jobs not marked as done; when used the jobfile argument should be the path '
                             'to the results folder of a previous execution of the script.')
    parser.add_argument('-l', '--jobs_limit', type=int, default=1000,
                        help='only submit jobs_limit jobs at a time.')
    parser.add_argument('-n', '--non_blocking', action='store_true',
                        help='preform a non blocking action if can.')

    args = parser.parse_args()

    run_str = 'RUN' if not args.rerun else 'RE-RUN'
    if args.dry_run:
        print(f'===== THIS IS A ***DRY*** {run_str} =====')
    else:
        print(f'===== THIS IS A ***REAL*** {run_str} =====')

    if not args.rerun:
        with open(args.jobfile, 'r') as f:
            job_description = yaml.safe_load(f)

        job_details = job_description['job_details']
        output_dir = job_details['output_dir']
        batch_name = datetime.datetime.now().strftime('%y-%m-%d') + '-' + job_details['name']
        batch_dir = os.path.join(output_dir, batch_name)
    else:
        jobfile = os.path.join(args.jobfile, 'job.yaml')
        with open(jobfile, 'r') as f:
            job_description = yaml.safe_load(f)
        job_details = job_description['job_details']
        batch_dir = args.jobfile
        batch_name = os.path.split(batch_dir)[-1]

    if os.path.exists(batch_dir) and os.listdir(batch_dir):
        if not (args.overwrite or args.rerun):
            raise FileExistsError('Directory exists and overwrite flag is not set')
        if args.overwrite:
            print('Removing existing output directory')
            shutil.rmtree(batch_dir)

    print(f'Writing experiment result to directory {batch_dir}')

    # --------------------- Find running jobs -------------------------------
    if args.rerun:
        running = set()
        squeue_out = subprocess.run('squeue -u $USER -o "%i %o"', shell=True, check=True, capture_output=True)
        squeue_out = str(squeue_out).split('\\n')
        squeue_out = squeue_out[1:-1]

        for line in squeue_out:
            ids, path = line.split()
            path = path[:path.rfind('submission_file')]

            if len(ids.split('_')) == 2:
                job_id, item_id = ids.split('_')
                if item_id[0] == '[' :
                    item_id = item_id[1:-1]
                    if '%' in item_id:
                        item_id, length = item_id.split('%')
                        length = int(length) # length is number of total sub-jobs

                    item_id_arr = []
                    for r in item_id.split(','):
                        start, end = r.split('-')
                        start, end = int(start), int(end)
                        item_id_arr.append( np.arange(start, end+1) )
                    item_id_arr = np.concatenate( item_id_arr )
                else:
                    item_id_arr = [item_id]
                
                full_ids = map(lambda x: "_".join( [job_id, str(x)] ), item_id_arr)
            else:
                assert len(ids.split('_')) == 1
                full_ids= [ids]


            for full_id in full_ids:
                full_path = path + full_id +'_submitted.pkl'
                with open(full_path, 'rb') as file:
                    job_detail = pickle.load(file)
                    running.add( job_detail.args[0] )

    # --------------------- HANDLE CONFIG ENUMERATION------------------------
    configs = [dict(c) for c in unwind_grid(deepcopy(job_description['parameters']))]
    configs_summary_str = grid_to_str(configs)
    configs = list(map(expand_tuple_keys, configs))
    total_configs = len(configs)
    nunique_keys = pd.DataFrame(configs).nunique(dropna=False)
    varying_keys = nunique_keys[nunique_keys > 1].index.values

    def grid_generator():
        for i, kvs in enumerate(configs):
            spec_name = '+'.join([f'{format_key(k)}={format_value(kvs[k])}'
                                 for k in varying_keys])
            spec_name = f'{i:03d}_{batch_name}+{spec_name}'
            out_dir = os.path.join(batch_dir, spec_name)
            if os.path.exists(os.path.join(out_dir, 'done')):
                continue
            spec = dict(output_dir=out_dir, **kvs)
            spec_filename = os.path.join(out_dir, 'spec.yaml')
            if not args.dry_run:
                os.makedirs(out_dir, exist_ok=True)
                with open(spec_filename, 'w') as f:
                    yaml.dump(spec, f, sort_keys=True)

            yield spec_name, spec_filename, out_dir

    def grid_generator_rerun():
        for spec_name in os.listdir(batch_dir):
            out_dir = os.path.join(batch_dir, spec_name)
            if spec_name == 'logs':
                continue
            if not os.path.isdir(out_dir) or spec_name.startswith('.'):
                continue
            if os.path.exists(os.path.join(out_dir, 'done')):
                continue
            spec_filename = os.path.join(out_dir, 'spec.yaml')
            if spec_filename in running:
                continue
            print(out_dir)
            yield spec_name, spec_filename, out_dir


    def get_spec_filename_array():
        spec_filename_array = []
        spec_name_array = []
        gen = grid_generator if not args.rerun else grid_generator_rerun
        for spec_name, spec_filename, out_dir in gen():
            spec_filename_array.append(spec_filename)
            spec_name_array.append(spec_name)
        return spec_filename_array, spec_name_array

    if args.rerun:
        spec_filename_array, spec_name_array = get_spec_filename_array()
        total_configs = len(spec_filename_array)

    print(f'Ready to run {total_configs} configurations: {configs_summary_str}')
    if not args.yes:
        input("Press Enter to continue...")

    if not args.rerun:
        spec_filename_array, spec_name_array = get_spec_filename_array() 

    # --------------------- HANDLE OUTPUT FOLDER -------------------------------
    if not (args.dry_run or args.rerun):
        os.makedirs(batch_dir, exist_ok=True)
        job_outfile = os.path.join(batch_dir, 'job.yaml')
        if os.path.exists(job_outfile):
            c = 1
            while os.path.exists(os.path.join(batch_dir, f'job.old.{c}.yaml')):
                c += 1
            os.rename(job_outfile,
                      os.path.join(batch_dir, f'job.old.{c}.yaml'))
            print(f'Renaming old job file to job.old.{c}.yaml')

        with open(job_outfile, 'w') as f:
            yaml.dump(job_description, f, default_flow_style=False)

    log_folder = batch_dir + "/logs"
    if not args.dry_run and not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)

    with open(args.script, 'r') as f:
        slurm_config = yaml.safe_load(f)
    slurm_config = Slurm_Config(**slurm_config)
    slurm_config.job_name = job_details['name']

    executor = submitit.SlurmExecutor(folder=log_folder)
    executor.update_parameters(**asdict(slurm_config))

    if args.dry_run:
        cmd = executor._submitit_command_str
        cmd = executor._make_submission_file_text(cmd, "[JOB_UID]")
        print(f'Would now run "{cmd}"')
    else: 
        count = 0
        while count < len(spec_filename_array):
            length = min(len(spec_filename_array)-count, args.jobs_limit)

            loop = True
            tries = 0
            while loop:
                loop = False
                try:
                    jobs = executor.map_array(call_main_with_args, spec_filename_array[count:count+length], spec_name_array[count:count+length], [batch_dir]*length)
                except submitit.core.utils.FailedJobError as error:
                    if 'QOSMaxSubmitJobPerUserLimit' in str(error) and tries < 5 and not args.non_blocking:
                        sleep(30)
                        loop = True
                        tries += 1
                    else:
                        raise error

            print(f'\nStarted {len(jobs)} jobs')
            if (count + length < len(spec_filename_array)) or not args.non_blocking:
                try:
                    outputs = [job.result() for job in jobs]

                except submitit.core.utils.UncompletedJobError as error:
                    print("got:")
                    print(error)
                    sleep(30)

                print(f'Finished {len(jobs)} jobs')
            count += length

    
    count = len(spec_filename_array)
    print(f'\nFinished running script')




