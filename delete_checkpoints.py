import os
import re

# List of directories to iterate over
PATHS = [
    '24-03-17-l=3_d=96', '24-03-17-l=3_d=96', '24-03-17-l=3_d=96_smaller_batch',
    '24-03-17-l=5_d=160_smaller_batch', '24-03-17-l=5_d=160_smaller_batch_correct_lr',
    '24-03-17-l=3_d=96_extended', '24-03-17-l=5_d=160_extended', '24-03-17-l=5_d=160_extended_2',
    '24-03-17-l=9_d=256', '24-03-17-l=12_d=352_single_node', '24-03-17-l=12_d=352_single_node_2',
    '24-03-17-l=18_d=544_single_node', '24-03-18-l=12_d=352_extended', '24-03-18-l=18_d=544_extended',
    '24-03-18-l=18_d=544_two_nodes', '24-03-18-l=22_d=640_two_nodes'
]

tuned_lr = {99614720: {'lr': 0.021, 'wd': 0.3},
 201326592: {'lr': 0.03, 'wd': 0.1},
 205520896: {'lr': 0.01, 'wd': 0.03},
 218103808: {'lr': 0.021, 'wd': 0.1},
 364904448: {'lr': 0.01, 'wd': 0.1},
 402653184: {'lr': 0.01, 'wd': 0.03},
 411041792: {'lr': 0.003, 'wd': 0.1},
 436207616: {'lr': 0.01, 'wd': 0.1},
 729808896: {'lr': 0.003, 'wd': 0.03},
 805306368: {'lr': 0.01, 'wd': 0.1},
 822083584: {'lr': 0.01, 'wd': 0.1},
 924844032: {'lr': 0.003, 'wd': 0.03},
 1417674752: {'lr': 0.001, 'wd': 0.01},
 1459617792: {'lr': 0.01, 'wd': 0.03},
 1644167168: {'lr': 0.01, 'wd': 0.1},
 1853882368: {'lr': 0.01, 'wd': 0.1},
 2835349504: {'lr': 0.01, 'wd': 0.003},
 2919235584: {'lr': 0.001, 'wd': 0.01},
 3707764736: {'lr': 0.003, 'wd': 0.03},
 5670699008: {'lr': 0.003, 'wd': 0.01},
 7415529472: {'lr': 0.03, 'wd': 0.01}}

checkpoints_to_keep_dict = {
    8: [8,6,4,2,1],
    16: [16,12,8,4,2],
    32: [32,24,16,8,4],
    64: [64,32,16,8,4],
}

checkpoint_sizes = {
    'l=3_d=96': 0,
    'l=5_d=160': 0,
    'l=9_d=256': 0,
    'l=12_d=352': 0,
    'l=18_d=544': 0,
    'l=22_d=640': 0,
}
def delete_checkpoints(base_path):
    total_size_of_checkpoints = 0
    total_size_of_deleted_checkpoints = 0
    for path in PATHS:
        full_path = os.path.join(base_path, path)
        if not os.path.exists(full_path):
            print(f"Directory does not exist: {full_path}")
            continue

        # Iterate over all sub-folders in each directory
        for sub_folder in os.listdir(full_path):
            sub_folder_path = os.path.join(full_path, sub_folder)
            if not os.path.isdir(sub_folder_path):
                continue
            
            # Check if the sub-folder name contains 'epo='
            match = re.search(r'epo=(\d+)', sub_folder)
            if match or 'l=3_d=96' in sub_folder or 'l=5_d=160' in sub_folder:
                if 'l=3_d=96_smaller' in sub_folder:
                    x = 32
                elif 'l=3_d=96' in sub_folder:
                    x = 16
                elif 'l=5_d=160' in sub_folder:
                    # read epoch number from job.yaml
                    with open(os.path.join(sub_folder_path, 'args.yaml'), 'r') as file:
                        for line in file:
                            match = re.search(r'epochs: (\d+)', line)
                            if match:
                                x = int(match.group(1))
                else:   
                    x = int(match.group(1))
                # print('going over sub-folder: ', sub_folder ,'in path: ', full_path)
                with open(os.path.join(sub_folder_path, 'args.yaml'), 'r') as file:
                    for line in file:
                        tokens = re.search(r'train_num_samples: (\d+)', line)
                        if tokens:
                            max_tokens = int(tokens.group(1)) * 2048 * x
                        # the whole line should be 'lr: 0.01' or 'wd: 0.1', for example. then we extract the number
                        lr_match = re.search(r'lr: (\d+\.\d+)', line)
                        if lr_match:
                            lr = float(lr_match.group(1))
                        wd_match = re.search(r'wd: (\d+\.\d+)', line)
                        if wd_match:
                            wd = float(wd_match.group(1))
                checkpoint_threshold = x - (x//8) - 1
                checkpoint_dir = os.path.join(sub_folder_path, 'checkpoints')
                
                if os.path.exists(checkpoint_dir):
                    for file in os.listdir(checkpoint_dir):
                        file_path = os.path.join(checkpoint_dir, file)
                        size_file = os.path.getsize(file_path)
                        total_size_of_checkpoints += size_file
                        # Extract number from filename and delete if condition is met
                        file_match = re.search(r'(\d+)', file)
                        if file_match:
                            file_number = int(file_match.group(1))
                            # if file_number not in checkpoints_to_keep_dict[x]:
                            nearest_key = min(tuned_lr.keys(), key=lambda k: abs(k - max_tokens))
                            # print(f'lr is {lr} and wd is {wd} and nearest key is {nearest_key} and tuned_lr[nearest_key]["lr"] is {tuned_lr[nearest_key]["lr"]} and tuned_lr[nearest_key]["wd"] is {tuned_lr[nearest_key]["wd"]}')
                            if file_number not in checkpoints_to_keep_dict[x]:
                                os.remove(os.path.join(checkpoint_dir, file))
                                total_size_of_deleted_checkpoints += size_file
                                # print(f"Deleted: {file} in {checkpoint_dir}")
                        # if size_file not in checkpoint_sizes.values() and 'epoch' in file_path:
                        for k in checkpoint_sizes.keys():
                            if k in path and 'epoch' in file_path:
                                checkpoint_sizes[k] = size_file

    print(f"Total size of checkpoints: {total_size_of_checkpoints / 1024 / 1024 / 1024} GB")
    print(f"Total size of deleted checkpoints: {total_size_of_deleted_checkpoints / 1024 / 1024 / 1024} GB")
    #  print a table of checkpoint sizes
    print("Checkpoint sizes:")
    for key, value in checkpoint_sizes.items():
        print(f"{key}: {value / 1024 / 1024} MB")

delete_checkpoints('exps')
