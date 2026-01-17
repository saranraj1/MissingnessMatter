import os

with open('debug_output_py.txt', 'w', encoding='utf-8') as log:
    log.write(f"CWD: {os.getcwd()}\n")
    raw_dir = 'data/raw'
    if os.path.exists(raw_dir):
        files = os.listdir(raw_dir)
        log.write(f"Files in {raw_dir}:\n")
        for f in files:
            log.write(f"'{f}' - len: {len(f)}\n")
            
        target = 'pima-indians-diabetes.data.csv'
        if target in files:
            log.write(f"Found exact match: {target}\n")
        else:
            log.write(f"Did NOT find exact match: {target}\n")
    else:
        log.write(f"{raw_dir} does not exist\n")
