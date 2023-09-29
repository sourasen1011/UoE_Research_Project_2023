from time_invariant_surv import Time_Invariant_Survival
from time_variant_surv import Time_Variant_Survival
from traditional_models import CPH, AFT, RSF
from other_nn_models import PYC, DSM
from general_utils import *
from model_utils import *
from losses import *
from models import *

def run_fitters(config_file_path , data_folder_path , verbose = True):
    '''
    helper function to run time invariant survival
    speak : verbose or not
    '''
    # have a dictionary to store all the metrics
    eval_dict = {
        'tvs':{'cindex':0 ,'ibs':0},
        'tis':{'cindex':0 ,'ibs':0},
        'cph':{'cindex':0 ,'ibs':0},
        'aft':{'cindex':0 ,'ibs':0},
        'rsf':{'cindex':0 ,'ibs':0},
        'pyc':{'cindex':0 ,'ibs':0},
        'dsm':{'cindex':0 ,'ibs':0}
        }
    
    # Get configs
    with open(config_file_path, "r") as file:
        configs = json.load(file)

    # Read the pickled DataFrames
    with open(data_folder_path+'x_train.pickle', 'rb') as file:
        x_train = pickle.load(file)
    with open(data_folder_path+'x_test.pickle', 'rb') as file:
        x_test = pickle.load(file)
    with open(data_folder_path+'x_val.pickle', 'rb') as file:
        x_val = pickle.load(file)

    # Read the pickled DataFrames
    with open(data_folder_path+'x_train_reshape_tv.pickle', 'rb') as file:
        x_train_reshape_tv = pickle.load(file)
    with open(data_folder_path+'x_test_reshape_tv.pickle', 'rb') as file:
        x_test_reshape_tv = pickle.load(file)
    with open(data_folder_path+'x_val_reshape_tv.pickle', 'rb') as file:
        x_val_reshape_tv = pickle.load(file)

    # Read the pickled targets
    with open(data_folder_path+'y_train.pickle', 'rb') as file:
        y_train = pickle.load(file)
    with open(data_folder_path+'y_test.pickle', 'rb') as file:
        y_test = pickle.load(file)
    with open(data_folder_path+'y_val.pickle', 'rb') as file:
        y_val = pickle.load(file)

    #-----------------------------------------------------------------------------------------------
    # instantiate - Time Variant Survival
    tvs = Time_Variant_Survival(
            configs = configs, 
            x_train_reshape_tv = x_train_reshape_tv,
            x_test_reshape_tv = x_test_reshape_tv, 
            x_val_reshape_tv = x_val_reshape_tv,
            y_train = y_train,
            y_test = y_test,
            y_val = y_val
        )

    # fit
    tvs.fit(verbose = verbose)
    mean_ , up_ , low_ , y_test_dur , y_test_event = tvs.predict() # Visualize -> tis.visualize(mean_ , up_ , low_ , _from = 40 , _to = 50 )
    tvs_cindex , tvs_ibs = tvs.evaluation(mean_ , y_test_dur , y_test_event, plot = False)

    # populate corresponding values in eval dict
    eval_dict['tvs']['cindex'] = tvs_cindex
    eval_dict['tvs']['ibs'] = tvs_ibs

    #-----------------------------------------------------------------------------------------------
    # instantiate - PyCox
    pyc = PYC(configs = configs, train_data = x_train, test_data = x_test, val_data = x_val, num_durations = 10)

    # fit
    pyc.fit()

    # eval
    pyc_cindex , pyc_ibs = pyc.eval()
        
    # populate corresponding values in eval dict
    eval_dict['pyc']['cindex'] = pyc_cindex
    eval_dict['pyc']['ibs'] = pyc_ibs

    #-----------------------------------------------------------------------------------------------
    # instantiate - Deep Survival Machines
    dsm = DSM(configs = configs, train_data = x_train, test_data = x_test, val_data = x_val, num_durations = 10)

    # fit
    dsm.fit()

    # eval
    dsm_cindex , dsm_ibs = dsm.eval()
       
    # populate corresponding values in eval dict
    eval_dict['dsm']['cindex'] = dsm_cindex
    eval_dict['dsm']['ibs'] = dsm_ibs
    
    #-----------------------------------------------------------------------------------------------
    # instantiate - Time Invariant Survival
    tis = Time_Invariant_Survival(
        configs = configs, 
        train_data = x_train,
        test_data = x_test, 
        val_data = x_val
    )

    # fit
    tis.fit(verbose = verbose)
    mean_ , up_ , low_ , y_test_dur , y_test_event = tis.predict() # Visualize -> tis.visualize(mean_ , up_ , low_ , _from = 40 , _to = 50 )
    tis_cindex , tis_ibs = tis.evaluation(mean_ , y_test_dur , y_test_event, plot = False)
    
    # populate corresponding values in eval dict
    eval_dict['tis']['cindex'] = tis_cindex
    eval_dict['tis']['ibs'] = tis_ibs
    
    # -----------------------------------------------------------------------------------------------
    # instantiate - CPH
    cph = CPH(configs = configs, train_data = x_train, test_data = x_test, val_data = x_val)

    # fit
    cph.fit()
    # eval
    cph_cindex , cph_ibs = cph.eval(fitter_is_rsf = False)
        
    # populate corresponding values in eval dict
    eval_dict['cph']['cindex'] = cph_cindex
    eval_dict['cph']['ibs'] = cph_ibs

    #-----------------------------------------------------------------------------------------------
    # instantiate - AFT
    aft = AFT(configs = configs, train_data = x_train, test_data = x_test, val_data = x_val)

    # fit
    aft.fit()
    # eval
    aft_cindex , aft_ibs = aft.eval(fitter_is_rsf = False)
        
    # populate corresponding values in eval dict
    eval_dict['aft']['cindex'] = aft_cindex
    eval_dict['aft']['ibs'] = aft_ibs

    #-----------------------------------------------------------------------------------------------
    # instantiate - RSF
    rsf = RSF(configs = configs, train_data = x_train, test_data = x_test, val_data = x_val)

    # fit
    rsf.fit()
    # eval
    rsf_cindex , rsf_ibs = rsf.eval(fitter_is_rsf = True)
        
    # populate corresponding values in eval dict
    eval_dict['rsf']['cindex'] = rsf_cindex
    eval_dict['rsf']['ibs'] = rsf_ibs
    
    #-----------------------------------------------------------------------------------------------

    return eval_dict

if __name__ == "__main__":
    eval_dict = run_fitters()
    print(eval_dict)