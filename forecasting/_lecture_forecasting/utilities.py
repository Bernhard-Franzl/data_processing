import os
import argparse

def check_log_dir(log_dir, n_run):
    
    if os.path.exists(log_dir):
        print(f"{log_dir} - already exists")
        
        path_to_run = os.path.join(log_dir, f"run_{n_run}")
        if os.path.exists(path_to_run):
            print(f"run_{n_run} - already exists")
            
            max_comb = max(list(map(lambda x: int(x.split("_")[-1]), os.listdir(path_to_run))))
            print(f"max combination number - {max_comb}")
            
            start_comb = max_comb + 1
            
        else:
            print(f"run_{n_run} - does not exist")
            start_comb = 0
            
    else:
        print(f"{log_dir} - does not exist")
        start_comb = 0
    
    print(f"Starting combination number: {start_comb}\n")
    return start_comb
        
def avoid_name_conflicts(tb_log_dir, cp_log_dir, run_number):
    
    tb_start_comb = check_log_dir(tb_log_dir, run_number)
    
    cp_start_comb = check_log_dir(cp_log_dir, run_number)

    if tb_start_comb != cp_start_comb:
        raise ValueError("Start combination numbers do not match")
    else:
        return tb_start_comb
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_run", action="store", type=int, nargs='?')  # make argument optional
    parser.add_argument("n_param", action="store", type=int, nargs='?')  # make argument optional

    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This is raised when required arguments are not provided.
        print("Some arguments were missing.")
        args = argparse.Namespace(n_run=None, n_param=None)

    return args

def prompt_for_missing_arguments(args):
    if args.n_run is None:
        args.n_run = int(input("Please provide the number of runs (n_run): "))

    if args.n_param is None:
        args.n_param = int(input("Please provide the number of parameters (n_param): "))

    return args
    
    
    