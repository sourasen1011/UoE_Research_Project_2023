from general_utils import *

# NN
from pycox.datasets import metabric
from pycox.models import LogisticHazard
import torchtuples as tt
from auton_survival.models.dsm import DeepSurvivalMachines

class _nn_fitter:
    '''
    simple wrapper class for NN models
    '''
    def __init__(self , configs , train_data , test_data , val_data , num_durations):
        self.configs = configs
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        # some aux vars
        self.num_durations = num_durations
        self.labtrans = LogisticHazard.label_transform(self.num_durations)

        # targets
        self.y_train = self.labtrans.fit_transform(*get_target(self.train_data))
        self.y_val = self.labtrans.transform(*get_target(self.val_data))
        self.out_features = self.labtrans.out_features

        # state var
        self.fitted = False
        self.fitter = None
    
    
    def eval(self):
        if not self.fitted:
            raise Exception('Model not fitted yet!')

        # _surv = self.fitter.predict_surv_df(self.test_data.iloc[: , :-2].to_numpy().astype('float32'))
        ev = EvalSurv(pd.DataFrame(self._surv), self.test_data['time_to_event'].to_numpy(), self.test_data['death'].to_numpy(), censor_surv='km')

        # get time grid
        time_grid_div = self.configs['time_invariant']['eval']['time_grid_div']
        time_grid = np.linspace(self.test_data['time_to_event'].min(), self.test_data['time_to_event'].max(), time_grid_div)
        # get metrics
        cindex = ev.concordance_td('antolini')
        ibs = ev.integrated_brier_score(time_grid)
        return cindex , ibs

class PYC(_nn_fitter):
    '''
    simple class for pycox logisitc hazards model
    '''
    def __init__(self , configs , train_data , test_data , val_data , num_durations):
        super(PYC , self).__init__(configs , train_data , test_data , val_data , num_durations)

    def fit(self):
        in_features = self.train_data.iloc[: , :-2].shape[1]
        num_nodes = [256,256]

        batch_norm = True
        dropout = 0.5

        train = (self.train_data.iloc[: , :-2].to_numpy().astype('float32'), self.y_train)
        val = (self.val_data.iloc[: , :-2].to_numpy().astype('float32'), self.y_val)

        net = tt.practical.MLPVanilla(in_features, num_nodes, self.out_features, batch_norm, dropout)

        model = LogisticHazard(net, tt.optim.Adam(0.002), duration_index=self.labtrans.cuts)

        batch_size = 256
        epochs = 500
        callbacks = [tt.cb.EarlyStopping()]

        log = model.fit(self.train_data.iloc[:,:-2].to_numpy().astype('float32'), self.y_train, batch_size, epochs, callbacks, val_data=val)
        
        # assign the fitted model to a class attr
        self.fitter = model
        # change state var
        self.fitted = True
        
        # predict
        self._surv = self.fitter.predict_surv_df(self.test_data.iloc[: , :-2].to_numpy().astype('float32'))

class DSM(_nn_fitter):
    '''
    simple class for DSM model
    '''
    def __init__(self , configs , train_data , test_data , val_data , num_durations):
        super(DSM , self).__init__(configs , train_data , test_data , val_data , num_durations)

    def fit(self):
        times = list(self.labtrans.cuts)

        param_grid = {'k' : [3,4],
              'distribution' : ['LogNormal'],
              'learning_rate' : [1e-3],
              'layers' : [[100],[100,100]]
             }

        params = ParameterGrid(param_grid)

        models = []
        for param in params:
            model = DeepSurvivalMachines(k = param['k'],
                                        distribution = param['distribution'],
                                        layers = param['layers'])
            # The fit method is called to train the model
            model.fit(self.train_data.iloc[: , :-2].to_numpy(), self.train_data['time_to_event'].to_numpy(), self.train_data['death'].to_numpy() ,
                    iters = 100 , 
                    learning_rate = param['learning_rate']
                    )
            models.append(
                [
                    [
                        model.compute_nll(self.val_data.iloc[: , :-2].to_numpy(), self.val_data['time_to_event'].to_numpy(), self.val_data['death'].to_numpy()), 
                        model,
                        param
                    ]
                ]
            )
        best_model = min(models)
        model = best_model[0][1]
        param = best_model[0][2]
        self.best_param = param
        
        # assign the fitted model to a class attr
        self.fitter = model
        # change state var
        self.fitted = True
        
        # predict
        out_survival = model.predict_survival(self.test_data.iloc[: , :-2].to_numpy().astype('float64'), times)
        self._surv = out_survival.T