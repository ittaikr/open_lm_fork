import os
import csv
import json
import numpy as np

def parse_evals(base_path):
    evals = []
    # for each jsonl file in the 'eval_results' directory
    # add a dictionary to the evals list
    # if the jsonl file contains more than one line, add a dictionary for the first line only
    # if the jsonl file is empty, skip it

    # also, if the jsonl file has key "loss" in it, add the mean of the np array to the dictionary
    # also add "loss_std" key to the dictionary with the standard deviation of the np array

    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if not 'eval_results' in subdir_path:
            continue
        for file in os.listdir(subdir_path):
            if not file.endswith('.jsonl'):
                continue
            file_path = os.path.join(subdir_path, file)
            dict_to_add = {}
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    continue
                dict_to_add = json.loads(lines[0])
                if 'loss' in dict_to_add:
                    dict_to_add['loss_std'] = np.array(dict_to_add['loss']).std()
                    dict_to_add['loss'] = np.array(dict_to_add['loss']).mean()
                    
                dict_to_add['tokens'] = max(dict_to_add['tokens'])
                evals.append(dict_to_add)
    
    # write the evals list to a csv file called 'summary_eval.csv' in the base_path directory
    with open(os.path.join(base_path, 'summary_eval.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=evals[0].keys())
        writer.writeheader()
        for eval in evals:
            writer.writerow(eval)
    
def check_if_evals_done(base_path):
    # check the number of jsonl files in the 'eval_results' directory (if exists)
    # compare it to the number of files in 'checkpoints' directory that have 'flop' in their name
    # if the number of jsonl files is equal to the number of 'flop' files, return True
    # otherwise, return False
    is_eval_results_subdir = ['eval_results' in subdir_path for subdir_path in os.listdir(base_path)]
    if not any(is_eval_results_subdir):
        # print("no eval_results")
        return -1
    for subdir in os.listdir(base_path):
        if not os.path.isdir(os.path.join(base_path, subdir)):
            continue
        if 'checkpoints' in subdir:
            continue
        subdir_path = os.path.join(base_path, subdir)
        jsonl_files = [file for file in os.listdir(subdir_path) if file.endswith('.jsonl')]
        checkpoints_path = os.path.join(base_path, 'checkpoints')
        flop_files = [file for file in os.listdir(checkpoints_path) if 'progress' not in file and 'optimizer' not in file]
        if len(jsonl_files) == len(flop_files):
            return 0
        else:
            print("in the middle", len(flop_files), len(jsonl_files))
            return len(flop_files) - len(jsonl_files)

def preform_evals(exps_path):
    not_done_count = 0
    for subdir in os.listdir(exps_path):
        subdir_path = os.path.join(exps_path, subdir)
        if '30-' in subdir or not os.path.isdir(subdir_path):
            continue
        for subsubdir in os.listdir(subdir_path):
            if not os.path.isdir(os.path.join(subdir_path, subsubdir)):
                continue
            subsubdir_path = os.path.join(subdir_path, subsubdir)
            not_done = check_if_evals_done(subsubdir_path)
            if not_done:
                # print(subsubdir, not_done)
                not_done_count += 1
            else:
                parse_evals(subsubdir_path)
    print("not done count:", not_done_count)

def main():
    # exps_path = 'exps_final_runs'
    # preform_evals(exps_path)
    sweep_path = 'exps_sweep'
    preform_evals(sweep_path)
    
if __name__ == '__main__':
    main()