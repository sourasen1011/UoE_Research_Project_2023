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
    
    def bin_duration(self , durations, cuts):
        '''
        simple function to discretize durations into buckets
        '''
        bin_edges = np.linspace(durations.min() , durations.max() , cuts)

        # Perform bucketing
        bucket_indices = np.digitize(durations, bin_edges)

        # Marker
        self.bin_dur = True

        return bucket_indices
    
    def modify_data(self , data , dur_col , eve_col , cuts):
        '''
        function to discretize duration
        '''
        data_mod = data.copy()

        # Apply binning of durations
        bucket_indices = self.bin_duration(durations = data_mod[dur_col] , cuts = cuts)
        data_mod[dur_col] = bucket_indices
        self.bucket_indices = bucket_indices #maintain the bucket indices object as an attribute of the class
        
        return data_mod.drop([dur_col , eve_col] , axis = 1) , data_mod[[dur_col , eve_col]].to_numpy()


