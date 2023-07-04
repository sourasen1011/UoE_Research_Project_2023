from scipy import interpolate
from datetime import timedelta
import numpy as np

class Patient_Image_Builder():
    '''
    Build patient image(s) from existing patient histor(y/ies).
    Remember to sort patient data as per charttime
    '''
    def __init__(self , data , cov_cols , resolution = 100 , subject_col = 'subject_id' , obs_time = 'charttime'):
        self.data = data
        self.resolution = resolution # what will be the granularity of the vital signs of the patient image?
        self.subject_col = subject_col
        self.obs_time = obs_time # observation times (referred to as charttime)
        self.data = self.data.sort_values([self.subject_col , self.obs_time]) # sort to maintain order
        self.cov_cols = cov_cols

    def build_img(self):
        '''
        Function to build patient image at given resolution
        '''
        patient_image = [] # container to hold all patient images
        counter = 0 #vestigial. can remove
        
        patient_list = self.data[self.subject_col].unique()
        print(f'unique patients: {len(patient_list)}')

        for pat in patient_list:
            _data = self.data[self.data[self.subject_col]==pat].sort_values(self.obs_time)
            _data['prev_'+self.obs_time] = _data[self.obs_time].shift(1)
            _data['time_diff'] = _data[self.obs_time] - _data['prev_'+self.obs_time]
            # for those who only have one record
            _data['time_diff'] = _data['time_diff'].fillna(timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0))

            # get time difference
            _data['time_diff'] = _data['time_diff'].dt.total_seconds().astype(int)

            # get cumulative time difference
            _data['cum_time_diff'] = _data['time_diff'].cumsum()
            
            # select necessary covariates
            _data = _data[self.cov_cols]

            # Backfill for covariates with NaN (but at least 1 non-NaN) - not clever but gets the job done
            _data = _data.bfill()

            _data = _data.fillna(0) # NEEDS IMMEDIATE REMEDY! WILL SKEW DATA!

            assert _data.isna().sum().sum() == 0 , 'Data contains nulls'

            # resolved observation times (evenly spaced between max and min observation time)
            resolved_observation_times = np.round(np.linspace(_data['cum_time_diff'].min() , _data['cum_time_diff'].max() , self.resolution))
            
            # container for storing single patient image
            time_resolved_covariates = []

            # interpolate at resolution provided
            for i , col in enumerate(_data.drop('cum_time_diff' , axis = 1).columns):
                f = interpolate.interp1d(_data['cum_time_diff'], _data[col])
                new_col = f(resolved_observation_times)
                # print(new_col.shape)
                time_resolved_covariates.append(new_col.reshape(-1 , 1))

            _resolved_data = np.hstack(time_resolved_covariates)

            patient_image.append(_resolved_data)
            
            counter += 1
            if counter%1000 == 0: print(f'{counter} patients done')
            # if counter>2000: break

        patient_image = np.stack(patient_image , axis = 0)

        return patient_image