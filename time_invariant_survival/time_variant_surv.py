from general_utils import *
from model_utils import *
from losses import *
from models import *


class Time_Variant_Survival:
    '''
    Class for fitting, testing, evaluating and explaining survival distributions
    '''
    def __init__(self, configs, x_train_reshape_tv, x_test_reshape_tv, x_val_reshape_tv, y_train, y_test, y_val):
        '''
        configs - dictionary created from loaded json file that contains all configs parsed from config file
        train_data - self.explanatory
        test_data - self.explanatory
        val_data - self.explanatory
        '''
        self.configs = configs
        # load patient images
        self.x_train_reshape_tv = x_train_reshape_tv
        self.x_test_reshape_tv = x_test_reshape_tv
        self.x_val_reshape_tv = x_val_reshape_tv
        # test
        assert self.x_train_reshape_tv.dim() == self.x_test_reshape_tv.dim() == self.x_val_reshape_tv.dim() == 4 , 'dimensions not correct'

        # load targets
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        # state vars
        self.fitted = False
        self.predicted = False
        
        # read from config file
        self.q_cuts = self.configs['time_variant']['training']['q_cuts']    # number of discretized durations
        self.hidden_size = self.configs['time_variant']['training']['hidden_size'] # hidden size of MLP
        self.output_size = self.q_cuts # same as discretizations
        self.alpha = self.configs['time_variant']['training']['alpha'] # trade off between two loss functions
        self.batch_size = self.configs['time_variant']['training']['batch_size'] # batch size for NN
        self.num_epochs = self.configs['time_variant']['training']['num_epochs'] # Number of epochs for NN
        self.learning_rate = self.configs['time_variant']['training']['learning_rate'] # LR for NN
        self.shuffle = self.configs['time_variant']['training']['shuffle'] # shuffle for Dataloader class
        self.patience = self.configs['time_variant']['training']['patience'] # patience for early stopping
        self.dropout = self.configs['time_variant']['training']['dropout'] # dropout for training and MC dropout  

    def fit(self , verbose = False):
        '''
        fitter function
        verbose: print on or off
        '''
        input_size = 7 * self.x_train_reshape_tv.shape[2] * self.x_train_reshape_tv.shape[3]
        
        # init loss
        l = generic_Loss()

        # init besst loss for early stopping
        best_loss = np.Inf

        # get features
        features = self.x_train_reshape_tv

        # get death time and event indicator
        y_train_dur , y_train_event = get_target(self.y_train)

        t_train = Transforms(durations = y_train_dur)
        dur_idx = t_train.discrete_transform(_cuts = self.q_cuts)
            
        # Create an instance of your custom dataset
        dataset = MyDataset(features, dur_idx , y_train_dur , y_train_event) # need to change outcomes[0] to indexed version
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle)    

        # build net
        self.net = Net(input_size , self.hidden_size , self.output_size , self.dropout)
        # init optim
        optimizer = torch.optim.Adam(self.net.parameters() , lr = self.learning_rate)
        
        # Prepare validation data
        # get duration, event
        y_val_dur , y_val_event = get_target(self.y_val)
        
        # transform to discrete
        t_val = Transforms(durations = y_val_dur)
        dur_idx_val = t_val.discrete_transform(_cuts = self.q_cuts)
                        
        # build surv matrix for val data
        sm_val = Surv_Matrix(duration_index = dur_idx_val , events = y_val_event , q_cuts = self.q_cuts)
        surv_mat_val = sm_val.make_survival_matrix()

        # Training loop
        for epoch in range(self.num_epochs):
            for batch_id , (patient_image , dur_idx , dur , eve) in enumerate(dataloader):
                # Prime for training
                self.net.train()
                    
                # forward pass
                phi_train = self.net(patient_image)

                # make survival matrix
                sm = Surv_Matrix(duration_index = dur_idx, events = eve , q_cuts = self.q_cuts)
                surv_mat = sm.make_survival_matrix()           

                # get loss
                loss_1 = l.nll_logistic_hazard(
                    logits = phi_train , 
                    targets = surv_mat , 
                    dur_idx = dur_idx
                    )
                loss_2 = l.c_index_lbo_loss(
                    logits = phi_train , 
                    times = dur , 
                    events = eve
                    )
                
                # combine
                loss = self.alpha*loss_1 + (1-self.alpha)*(loss_2)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                    
                # Early stopping
                with torch.no_grad():
                    # compute loss
                    phi_val = self.net(self.x_val_reshape_tv) # do we need to add torch.Tensor? might be redundant. 
                    val_loss_1 = l.nll_logistic_hazard(
                        logits = phi_val, 
                        targets = surv_mat_val , 
                        dur_idx = dur_idx_val
                        )
                    val_loss_2 = l.c_index_lbo_loss(
                        logits = phi_val , 
                        times = torch.Tensor(y_val_dur) , 
                        events = torch.Tensor(y_val_event)
                        )

                    # combine
                    val_loss = self.alpha*val_loss_1 + (1-self.alpha)*(val_loss_2)
                    
                # Check if validation loss has improved
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1

                # Check if early stopping condition is met
                if counter >= self.patience:
                    # print(f"Early stopping at epoch {epoch}.")
                    break
                
            # control verbosity
            if verbose:
                if ((epoch+1)%50==0): 
                    print(f"Epoch {epoch+1}: Training Loss: {loss.item():.7f}, Val Loss: {val_loss.item():.7f}") 
        
        # change state
        self.fitted = True

    def predict(self):
        '''
        this is the prediction suite
        '''       
        if not self.fitted:
            raise Exception("Model isn't fitted yet!")

        # Testing
        surv = [] # length will be equal to number of cluster
        mc_iter = self.configs['time_variant']['testing']['mc_iter']
        conf_int_lower = self.configs['time_variant']['testing']['conf_int_lower']
        conf_int_upper = self.configs['time_variant']['testing']['conf_int_upper']
        
        # get features, death time and event indicator
        features = self.x_test_reshape_tv # do we need to add torch.Tensor? might be redundant. 
            
        # get death time and event indicator
        y_test_dur , y_test_event = get_target(self.y_test)

        surv = []

        # apply Monte Carlo dropout
        for _ in range(mc_iter):
                
            # Prime dropout layers
            self.net.train()
                
            # predict
            mc_haz = torch.sigmoid(self.net(features))
            mc_survival = torch.cumprod(1 - mc_haz , dim = 1).detach().numpy()

            # append survivals from different runs
            surv.append(mc_survival)
            
        # convert to 3d array
        surv = np.array(surv)

        # get stats
        mean_ = np.mean(surv , axis = 0)
        up_ = np.quantile(surv , axis = 0 , q = conf_int_upper)
        low_ = np.quantile(surv , axis = 0 , q = conf_int_lower)

        # QCs
        assert mean_.shape[0] == up_.shape[0] == low_.shape[0] == y_test_dur.shape[0] == y_test_event.shape[0] , 'shape mismatch'

        # change
        self.predicted = True
        
        return mean_ , up_ , low_ , y_test_dur , y_test_event

    
    def visualize(self , mean_ , low_ , up_ , _from , _to):
        '''
        visualize the predictions
        '''
        # get features, death time and event indicator
        features = self.test_data

        # get death time and event indicator
        y_test_dur_ , y_test_event_ = get_target(features)

        t_test = Transforms(durations = y_test_dur_)
        dur_idx_test = t_test.discrete_transform(_cuts = self.q_cuts) # although we don't use the dur_idx_test variable,
        # we actually need the fitted t_test object

        # get transparency for graph
        transparency = self.configs['time_variant']['test_viz']['transparency']
        _ = plot_with_cf(t_test.bin_edges, mean_ , low_ , up_ , _from , _to , transparency = transparency)


    def evaluation(self , mean_ , y_test_dur , y_test_event , plot = False):
        '''
        Evaluation by
        1. td c-index
        2. Brier score
        3. IBS
        '''
        time_grid_div = self.configs['time_variant']['eval']['time_grid_div']
        time_grid = np.linspace(y_test_dur.min(), y_test_dur.max(), time_grid_div)
        
        # Evaluation
        ev_ = EvalSurv(pd.DataFrame(mean_.T) , y_test_dur , y_test_event , censor_surv='km')
        
        # brier score
        if plot:
            ev_.brier_score(time_grid).plot()
            plt.ylabel('Brier score')
            _ = plt.xlabel('Time')

        # td c-index
        tdci = ev_.concordance_td()
        # print(f'concordance-td: {tdci}')
        
        # IBS
        ibs = ev_.integrated_brier_score(time_grid)
        # print(f'integrated brier score {ibs}')
        
        return tdci , ibs