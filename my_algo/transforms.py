import numpy as np

class Transforms:
    '''
    simple class containing transformations
    '''
    def __init__(self):
        '''
        states of transformations
        '''
        self.bin_dur = False
        self.bin_edges = None
        self.bucket_indices = None
        self.n_bucket_indices = None
    
    def bin_duration(self , data , subject_col , dur_col , eve_col , cuts):
        '''
        Simple function to discretize durations into buckets
        Params -
        durations: np.array: survival/censoring times of patients
        cuts: int: specifies the number of divisions between minimum and maximum duration of cohort 
        '''
        # data attributes
        self.data = data[[subject_col , dur_col , eve_col]].drop_duplicates()
        print(f'number of patients: {len(self.data)}')
        # self.dur_col = dur_col
        # self.eve_col = eve_col

        bin_edges = np.linspace(self.data[dur_col].min()+1 , self.data[dur_col].max()-1 , cuts) # the -/+1 is there so that we can automatically get 0s and maxes
        self.bin_edges = bin_edges

        # Perform bucketing
        bucket_indices = np.digitize(self.data[dur_col], bin_edges)
        bucket_indices = bucket_indices

        # broadcast subtract 1 to maintain starting from zero <--- deprecated comment

        # Marker
        self.bin_dur = True
        self.bucket_indices = bucket_indices #maintain the bucket indices object as an attribute of the class
        self.n_bucket_indices = len(np.unique(self.bucket_indices)) #maintain number of buckets

        # returns
        pats , dur_idx , events = self.data[subject_col] , self.bucket_indices , self.data[eve_col]
        return np.stack([np.array(pats) , np.array(dur_idx) , np.array(events)] , axis = 1)
        
    
    # def modify_data(self , durations , events , cuts):
    #     '''
    #     function to discretize duration
    #     '''
    #     # data_mod = data.copy()

    #     # Apply binning of durations
    #     bucket_indices = self.bin_duration(durations = durations , cuts = cuts)
    #     # data_mod[dur_col] = bucket_indices
    #     self.bucket_indices = bucket_indices #maintain the bucket indices object as an attribute of the class
    #     self.n_bucket_indices = len(np.unique(self.bucket_indices)) #maintain number of buckets
        
    #     return data_mod.drop([dur_col , eve_col] , axis = 1) , data_mod[[dur_col , eve_col]].to_numpy().T # transpose is necessary

    # def modify_data(self , data , dur_col , eve_col , cuts):
    #     '''
    #     function to discretize duration
    #     '''
    #     data_mod = data.copy()

    #     # Apply binning of durations
    #     bucket_indices = self.bin_duration(durations = data_mod[dur_col] , cuts = cuts)
    #     data_mod[dur_col] = bucket_indices
    #     self.bucket_indices = bucket_indices #maintain the bucket indices object as an attribute of the class
    #     self.n_bucket_indices = len(np.unique(self.bucket_indices)) #maintain number of buckets
        
    #     return data_mod.drop([dur_col , eve_col] , axis = 1) , data_mod[[dur_col , eve_col]].to_numpy().T # transpose is necessary

