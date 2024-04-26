import os

def clean_exp(dir):
    # for all subdirectories in dir
    for subdir in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, subdir)):
            # remove 'checkpoints' directory in subdir
            if os.path.exists(os.path.join(dir, subdir, 'checkpoints')):
                os.system('rm -r ' + os.path.join(dir, subdir, 'checkpoints'))
                # print('rm -r ' + os.path.join(dir, subdir, 'checkpoints'))

def main():
    dirs_to_clean = [dir for dir in os.listdir('exps') if (os.path.isdir(os.path.join('exps', dir)) and '24-03-1' in dir)]
    for dir in dirs_to_clean:
        clean_exp(os.path.join('exps', dir))

if __name__ == '__main__':
    main()