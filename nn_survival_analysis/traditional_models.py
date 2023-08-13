from general_utils import *

# Traditional
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter

# Tree-Based
from sksurv.ensemble import RandomSurvivalForest

class _traditional_fitter:
    '''
    simple wrapper class for traditional and tree-based model(s)
    '''
    def __init__(self , configs , train_data , test_data , val_data):
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.configs = configs
        # state var
        self.fitted = False
        self.fitter = None

    def eval(self , fitter_is_rsf = False):
        if not self.fitted:
            raise Exception('Model not fitted yet!')
            
        # get evaluation - with a tweak for RSF. not the best. but works for now.
        if not fitter_is_rsf:
            # predict
            _surv = self.fitter.predict_survival_function(self.test_data.iloc[: , :-2])
            ev = EvalSurv(pd.DataFrame(_surv), self.test_data['time_to_event'].to_numpy(), self.test_data['death'].to_numpy(), censor_surv='km')
        else:
            # predict
            _surv = self.fitter.predict_survival_function(self.test_data.iloc[: , :-2] , return_array = True)
            ev = EvalSurv(pd.DataFrame(_surv.T), self.test_data['time_to_event'].to_numpy(), self.test_data['death'].to_numpy(), censor_surv='km')
        
        # get time grid
        time_grid_div = self.configs['time_invariant']['eval']['time_grid_div']
        time_grid = np.linspace(self.test_data['time_to_event'].min(), self.test_data['time_to_event'].max(), time_grid_div)
        # get metrics
        cindex = ev.concordance_td('antolini')
        ibs = ev.integrated_brier_score(time_grid)
        return cindex , ibs
    
class CPH(_traditional_fitter):
    '''
    simple class for cox proportional hazards model
    '''
    def __init__(self , configs , train_data , test_data , val_data):
        super(CPH , self).__init__(configs , train_data , test_data , val_data)

    def fit(self):
        # init CPH
        cph = CoxPHFitter(penalizer = 0.1)
        # fit
        cph.fit(self.train_data, duration_col='time_to_event', event_col='death', fit_options = {'step_size':0.1})
        # assign the fitted model to a class attr
        self.fitter = cph
        # change state var
        self.fitted = True

class AFT(_traditional_fitter):
    '''
    simple class for weibull accelerated failure times model
    '''
    def __init__(self , configs , train_data , test_data , val_data):
        super(AFT , self).__init__(configs , train_data , test_data , val_data)

    def fit(self):
        # init AFT
        aft = WeibullAFTFitter(penalizer = 0.01)
        eps = 1e-8
        self.train_data['time_to_event'] = self.train_data['time_to_event'] + eps
        # fit
        aft.fit(self.train_data, duration_col='time_to_event', event_col='death')

        # assign the fitted model to a class attr
        self.fitter = aft
        # change state var
        self.fitted = True   

class RSF(_traditional_fitter):
    '''
    simple class for random survival forest model
    '''
    def __init__(self , configs , train_data , test_data , val_data):
        super(RSF , self).__init__(configs , train_data , test_data , val_data)

    def fit(self):
        # Train - Create a structured array
        y_train = np.array([(x, y) for x, y in zip(self.train_data['death'].astype('bool') , self.train_data['time_to_event'])],
                                    dtype=[('death', bool) , ('time_to_event', int)])

        # init RSF
        rsf = RandomSurvivalForest(
            n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, oob_score = True
        )
        rsf.fit(self.train_data.iloc[: , :-2], y_train)

        # assign the fitted model to a class attr
        self.fitter = rsf
        # change state var
        self.fitted = True   