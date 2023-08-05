from time_invariant_surv import *

def run_tis():
    '''
    helper function to run time invariant survival
    '''
    # Get configs
    with open(config_file_path, "r") as file:
        configs = json.load(file)

    # Read the pickled DataFrames
    with open('../05_preprocessing_emr_data/data/x_train.pickle', 'rb') as file:
        x_train = pickle.load(file)
    with open('../05_preprocessing_emr_data/data/x_test.pickle', 'rb') as file:
        x_test = pickle.load(file)
    with open('../05_preprocessing_emr_data/data/x_val.pickle', 'rb') as file:
        x_val = pickle.load(file)

    # Read the pickled DataFrame
    with open('../05_preprocessing_emr_data/data/consolidated_pat_tbl.pickle', 'rb') as file:
        consolidated_pat_tbl = pickle.load(file)

    # instantiate
    tis = Time_Invariant_Survival(
        configs = configs, 
        train_data = x_train,
        test_data = x_test, 
        val_data = x_val
    )

    tis.fit(verbose = True)
    mean_ , up_ , low_ , y_test_dur , y_test_event = tis.predict()
    # obj.visualize(mean_ , up_ , low_ , _from = 40 , _to = 50 )
    cindex , ibs = tis.evaluation(mean_ , y_test_dur , y_test_event, plot = False)

    return cindex , ibs

if __name__ == "__main__":
    cindex , ibs = run_tis()
    print(cindex , ibs)