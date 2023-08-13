import time
import argparse
import sys
sys.path.append('../')
sys.path.append('../nn_survival_analysis')
from nn_survival_analysis.run_fitters import run_fitters

# define func for looped fitting
def repeat_fits(iter , verbose):
    '''
    iter: how many times will the model(s) be fit
    '''
    # have a dictionary to store all the metrics
    eval_dict_iter = {
            'tvs':{'cindex':[] , 'ibs':[]},
            'tis':{'cindex':[] , 'ibs':[]},
            'cph':{'cindex':[] , 'ibs':[]},
            'aft':{'cindex':[] , 'ibs':[]},
            'rsf':{'cindex':[] , 'ibs':[]},
            'pyc':{'cindex':[] , 'ibs':[]},
            'dsm':{'cindex':[] , 'ibs':[]}
            }    
    
    for i in range(iter):
        print(f"Running experiment iteration {i+1}")
        eval_dict = run_fitters(config_file_path =  '../nn_survival_analysis/config.json', verbose = verbose)
        # print(eval_dict)
        for model in eval_dict:
            for eval_metric in eval_dict[model]:
                eval_dict_iter[model][eval_metric].append(eval_dict[model][eval_metric])

    return eval_dict_iter

# main func
def main():
    # get time now
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run an experiment loop")
    parser.add_argument("--iter", type=int, default=1, help="Number of times to run the experiment loop")
    parser.add_argument("--verbose", type=bool, default=True, help="Controls verbosity")
    args = parser.parse_args()

    # get results
    res = repeat_fits(args.iter , args.verbose)

    # get ending time
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    print(res)

    return res

if __name__ == "__main__":
    main()
