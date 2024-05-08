import os

def clean_exp(dir):
    # for all subdirectories in dir
    for subdir in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, subdir)):
            # remove 'checkpoints' directory in subdir
            if os.path.exists(os.path.join(dir, subdir, 'checkpoints')):
                # os.system('rm -r ' + os.path.join(dir, subdir, 'checkpoints'))
                print('rm -r ' + os.path.join(dir, subdir, 'checkpoints'))

def check_nccl(dir):
    # for all subdirectories in dir
    for subdir in os.listdir(dir):
        # dubdirectory should be a directory, and contain files with <job_id>.out
        if os.path.isdir(os.path.join(dir, subdir)) and "done" not in os.listdir(os.path.join(dir, subdir)):
            files = os.listdir(os.path.join(dir, subdir))
            # we want to check to file with max job_id
            max_job_id = -1
            for file in files:
                if '.out' in file:
                    job_id = int(file.split('.')[0])
                    if job_id > max_job_id:
                        max_job_id = job_id
            file_to_check = str(max_job_id) + '.out'
            # check if file_to_check contains 'To avoid data inconsistency, we are taking the entire process down.'
            with open(os.path.join(dir, subdir, file_to_check), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'To avoid data inconsistency, we are taking the entire process down.' in line:
                        print(os.path.join(dir, subdir))
                        break

def main():
    dir_name = 'exps_final_runs'
    dirs_to_clean = [dir for dir in os.listdir(dir_name) if (os.path.isdir(os.path.join(dir_name, dir))) and ('24-05-07' not in dir) and ('chinchilla' not in dir)]
    for dir in dirs_to_clean:
        clean_exp(os.path.join(dir_name, dir))
    # dirs_to_check = [dir for dir in os.listdir('exps_final_runs')]
    # for dir in dirs_to_check:
    #     check_nccl(os.path.join('exps_final_runs', dir))
if __name__ == '__main__':
    main()