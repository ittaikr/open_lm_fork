import os
from open_lm.file_utils import pt_load
import pandas as pd
import ast

num_tar_files = sum(1 for f in os.listdir('/p/fastdata/mmlaion/lmdata_2/openwebtext2_tokenized') if f.endswith('.tar'))



def count_owt2_passes(path):
    # print(path)
    owt2 = pt_load(path, map_location='cpu')
    next_shard_per_source = owt2['next_shard_per_source'][0]
    return next_shard_per_source / num_tar_files

def traverse_dir(path):
    # for all subdirectories in path
    passes_dict = []
    max_passes = 0
    for root, dirs, files in os.walk(path):
        # check only the 'checkpoints' subdir, if it exists
        if 'checkpoints' in dirs:
            path = os.path.join(root, 'checkpoints')
            # ckpt_path should be the onlt file in the 'checkpoints' subdir that have 'epoch' in their name
            # if no such file exists, continue to the next subdirectory
            ckpt_path = [os.path.join(path, f) for f in os.listdir(path) if 'epoch' in f and 'eval' not in f]
            if not ckpt_path:
                continue
            ckpt_path = ckpt_path[0]
            # count the number of passes in the checkpoint file
            passes = count_owt2_passes(ckpt_path)
            passes_dict.append({
                'path': root,
                'passes': passes
            })
            if passes > max_passes:
                max_passes = passes

    return passes_dict, max_passes

def traverse_dir_root(root):
    # for all subdirectories in path
    passes_dict_root = []
    root_max_passes = 0
    for root, dirs, files in os.walk(root):
        # run traverse_dir on each subdirectory
        passes_dict, max_passes = traverse_dir(root)
        passes_dict_root.append(passes_dict)
        if max_passes > root_max_passes:
            root_max_passes = max_passes

    return passes_dict_root, root_max_passes

if __name__ == '__main__':
    path_to_csv = 'owt2_passes_clean_filtered.csv'
    

    # Read the CSV
    df = pd.read_csv(path_to_csv)

    # Initialize lists for paths and passes
    # paths = []
    # passes = []

    # for i in range(1, len(df.columns)):
    #     # for row in df:
    #     for j in range(len(df)):
    #         cell = df.iloc[j, i]
    #         if pd.notna(cell):
    #             data = ast.literal_eval(cell)
    #             paths.append(data['path'])
    #             passes.append(data['passes'])

    # # Create the new dataframe
    # result_df = pd.DataFrame({
    #     'path': paths,
    #     'passes': passes
    # })

    # result_df.to_csv('owt2_passes_clean.csv', index=False)

    # result_df.query('passes > 1').to_csv('owt2_passes_clean_filtered.csv', index=False)
    df['exp'] = df['path'].apply(lambda x: '/'.join(x.split('/')[:-1]))
    df.groupby('exp').count().to_csv('owt2_passes_clean_filtered_grouped.csv')

    