from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import DataFrameMapper
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns

# store data here
folder_path = "../data/"

# access creds
with open('../psql_key.json') as file:
    creds = json.load(file)

# Create an SQLAlchemy engine with the PostgreSQL database URI
engine = create_engine(f"postgresql+psycopg2://{creds['user']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['database']}")

# these diag codes were taken from table mimiciv_hosp.d_icd_diagnoses
hf_codes = '''('39891','40200','40201','40210','40211','40290','40291','40400','40401','40402','40403','40410','40411','40412',
'40413','40490','40491','40492','40493','4280','4281','42820','42821','42822','42823','42830','42831','42832','42833','42840','42841',
'42842','42843','4289','E8726','E8745','I0981','I110','I119','I130','I131','I1310','I1311','I132','I50','I501','I502','I5020','I5021','I5022',
'I5023','I503','I5030','I5031','I5032','I5033','I504','I5040','I5041','I5042','I5043','I508','I5081','I50810','I50811','I50812','I50813','I50814',
'I5082','I5083','I5084','I5089','I509','I9713','I97130','I97131','T8622','T8632','Y625')'''

# get all admissions related to HF codes_____________________________________________________________________________________________________________
query = f'''
select distinct a.subject_id , a.hadm_id , a.admittime , a.dischtime , p.gender , p.anchor_age , p.dod
from (select hadm_id from mimiciv_hosp.diagnoses_icd where icd_code in {hf_codes}) d
inner join (select * from mimiciv_hosp.admissions) a
on a.hadm_id = d.hadm_id
inner join (select * from mimiciv_hosp.patients) p
on p.subject_id = a.subject_id
'''

all_hf_pats_0 = pd.read_sql_query(query , engine)

all_hf_pats_0['dod'] = pd.to_datetime(all_hf_pats_0['dod']).dt.date
all_hf_pats_0['admittime'] = all_hf_pats_0['admittime'].dt.date
all_hf_pats_0['dischtime'] = all_hf_pats_0['dischtime'].dt.date

# QC
print(all_hf_pats_0['subject_id'].nunique())

# Get online medical record data (OMR)_____________________________________________________________________________________________________________
reqd_results = ("BMI (kg/m2)" , "Height (Inches)" , "Weight (Lbs)") # Not including BP as I can get it from vitalsigns  
q = f'''select distinct subject_id , chartdate , result_name , 
result_value::FLOAT from mimiciv_hosp.omr where result_name in {reqd_results}'''

omr = pd.read_sql_query(q , engine)
omr_first = omr.sort_values(['subject_id' , 'chartdate'] , ascending = True).groupby(['subject_id' , 'result_name']).first().reset_index()
omr_pivot = omr_first.pivot(index = 'subject_id', columns = 'result_name' , values = 'result_value').reset_index()

print(f'patients: {omr_pivot.subject_id.nunique()}')

# RACE_____________________________________________________________________________________________________________
q = '''
select subject_id , race , count(race) as c_race
from mimiciv_hosp.admissions
group by subject_id , race
order by c_race desc
'''

# Some patients have multiple race - chalking that up to a data discrepancy for now!
race_0 = pd.read_sql_query(q , engine)
race = race_0.groupby('subject_id').first().reset_index().drop('c_race' , axis = 1)

# QC
assert len(race) == race_0["subject_id"].nunique() , 'is not patient level table'

#QC 
assert race.groupby('subject_id').agg({'race':'nunique'}).sort_values('race' , ascending = False).iloc[0 , ][0] == 1 , 'multiple races'

race_broad = pd.DataFrame( [
    ('AMERICAN INDIAN/ALASKA NATIVE','NATIVE'),
    ( 'ASIAN','ASIAN'),
    ( 'ASIAN - ASIAN INDIAN','ASIAN'),
    ( 'ASIAN - CHINESE','ASIAN'),
    ( 'ASIAN - KOREAN','ASIAN'),
    ( 'ASIAN - SOUTH EAST ASIAN','ASIAN'),
    ( 'BLACK/AFRICAN','BLACK'),
    ( 'BLACK/AFRICAN AMERICAN','BLACK'),
    ( 'BLACK/CAPE VERDEAN','BLACK'),
    ( 'BLACK/CARIBBEAN ISLAND','BLACK'),
    ( 'HISPANIC OR LATINO','HISPANIC'),
    ( 'HISPANIC/LATINO - CENTRAL AMERICAN','HISPANIC'),
    ( 'HISPANIC/LATINO - COLUMBIAN','HISPANIC'),
    ( 'HISPANIC/LATINO - CUBAN','HISPANIC'),
    ( 'HISPANIC/LATINO - DOMINICAN','HISPANIC'),
    ( 'HISPANIC/LATINO - GUATEMALAN','HISPANIC'),
    ( 'HISPANIC/LATINO - HONDURAN','HISPANIC'),
    ( 'HISPANIC/LATINO - MEXICAN','HISPANIC'),
    ( 'HISPANIC/LATINO - PUERTO RICAN','HISPANIC'),
    ( 'HISPANIC/LATINO - SALVADORAN','HISPANIC'),
    ( 'MULTIPLE RACE/ETHNICITY','OTHER'),
    ( 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER','NATIVE'),
    ( 'OTHER','OTHER'),
    ( 'PATIENT DECLINED TO ANSWER','OTHER'),
    ( 'PORTUGUESE','WHITE'),
    ( 'SOUTH AMERICAN','HISPANIC'),
    ( 'UNABLE TO OBTAIN','OTHER'),
    ( 'UNKNOWN','OTHER'),
    ( 'WHITE','WHITE'),
    ( 'WHITE - BRAZILIAN','WHITE'),
    ( 'WHITE - EASTERN EUROPEAN','WHITE'),
    ( 'WHITE - OTHER EUROPEAN','WHITE'),
    ( 'WHITE - RUSSIAN','WHITE')
] , columns = ['race' , 'race_broad']
)
race = pd.merge(race , race_broad , on = 'race' , how = 'inner')
race.drop('race' , axis = 1 , inplace = True)
race.rename(columns = {'race_broad':'race'} , inplace = True)

print(f'patients {race.subject_id.nunique()}')

# LAB_____________________________________________________________________________________________________________
q = '''
select b.subject_id , b.itemid , avg(b.valuenum) as val
from
(select * from mimiciv_hosp.labevents a 
where a.itemid = 50907 
or a.itemid = 50983
or a.itemid = 51222
or a.itemid = 51133
-- or a.itemid = 51221
) b
group by b.subject_id , b.itemid
'''

bloodwork_0 = pd.read_sql_query(q , engine)

bloodwork = bloodwork_0.pivot(index = 'subject_id' , columns = 'itemid' , values = 'val').reset_index()
bloodwork.head()

# name change
bloodwork.rename(
    columns = {
        50907:'cholesterol' , 
        50983:'sodium' , 
        51133:'lymphocyte' , 
        51222:'hemoglobin',
        # 51221:'hematocrit'
        } , 
        inplace = True
    )

# QC
assert len(bloodwork) == bloodwork_0["subject_id"].nunique() , 'is not patient level table'

print(f'patients: {bloodwork.subject_id.nunique()}')

print(bloodwork.columns)

# MEDICATION_____________________________________________________________________________________________________________
hadm_id_list = "('"+"','".join(list(all_hf_pats_0['hadm_id'].unique().astype('str')))+"')"
hadm_id_list

q = f'''
select a.* from
(
    select * , lower(medication) as l_medication from mimiciv_hosp.emar
    where  hadm_id::VARCHAR in {hadm_id_list}
) a
where a.l_medication in ('amlodipine','diltiazem','felodipine','isradipine','nicardipine',
'nifedipine','nisoldipine','verapamil','azilsartan','candesartan','eprosartan','irbesartan',
'losartan','olmesartan','telmisartan','valsartan','benazepril','captopril','enalapril','fosinopril',
'lisinopril','moexipril','perindopril','quinapril','ramipril','trandolapril','acebutolol','atenolol',
'bisoprolol','metoprolol','nadolol','nebivolol','propranolol')
and a.event_txt = 'Administered'
'''

medication = pd.read_sql_query(q , engine)
# print(f'{len(medication) , medication.subject_id.nunique()}')

# cast to lower to avoid duplication
medication['medication'] = medication['medication'].str.lower()

# pivot
medication_pivot = medication.pivot_table(index = 'subject_id' , columns = 'medication' , values = 'hadm_id' , aggfunc = 'count').fillna(0).reset_index()

# QC
assert len(medication_pivot) == medication["subject_id"].nunique() , 'is not patient level table'

# QC
_df = pd.DataFrame(medication_pivot.sum(axis = 1) , columns = ['check'])

assert len(_df[_df['check'] == 0])==0 # checks if at least one patient has taken none of these drugs!

print(f'patients -> {medication_pivot.subject_id.nunique()}')

# Vital signs_____________________________________________________________________________________________________________

query = f'''
select subject_id , temperature , heartrate , resprate , o2sat , sbp , dbp from mimiciv_ed.vitalsign
'''
vitalsign_0 = pd.read_sql_query(query , engine)

vitalsign = vitalsign_0.groupby('subject_id').mean().reset_index()

# QC
assert len(vitalsign) == vitalsign_0["subject_id"].nunique() , 'is not patient level table'

print(f'patients -> {vitalsign.subject_id.nunique()}')

# Comorbidities_____________________________________________________________________________________________________________
comorbidity_dict = {
    'hypertension':['G932','H4005','H40051','H40052','H40053','H40059','I10','I15','I150','I158','I159','I270','I272','I2720',
                    'I2721','I2722','I2723','I2724','I2729','I873','I8730','I87301','I87302','I87303','I87309','I8731','I87311',
                    'I87312','I87313','I87319','I8732','I87321','I87322','I87323','I87329','I8733','I87331','I87332','I87333',
                    'I87339','I8739','I87391','I87392','I87393','I87399','I973','K766','O10','O100','O1001','O10011','O10012',
                    'O10013','O10019','O1002','O1003','O104','O1041','O10411','O10412','O10413','O10419','O1042','O1043','O109',
                    'O1091','O10911','O10912','O10913','O10919','O1092','O1093','O11','O111','O112','O113','O114','O115','O119',
                    'O12','O13','O131','O132','O133','O134','O135','O139','O16','O161','O162','O163','O164','O165','O169','P292',
                    'P2930','R030'],
    'obesity':['E66', 'E6601', 'E6609', 'E661', 'E662', 'E668'],
    'diabetes':['E101','E09','E090','E0900','E0901','E091','E0910','E0911','E092','E0921','E0922','E0929','E093','E0931','E09311',
                'E09319','E0932','E09321','E093211','E093212','E093213','E093219','E09329','E093291','E093292','E093293','E093299',
                'E0933','E09331','E093311','E093312','E093313','E093319','E09339','E093391','E093392','E093393','E093399','E0934',
                'E09341','E093411','E093412','E093413','E093419','E09349','E093491','E093492','E093493','E093499','E0935','E09351',
                'E093511','E093512','E093513','E093519','E09352','E093521','E093522','E093523','E093529','E09353','E093531','E093532',
                'E093533','E093539','E09354','E093541','E093542','E093543','E093549','E09355','E093551','E093552','E093553','E093559',
                'E09359','E093591','E093592','E093593','E093599','E0936','E0937','E0937X1','E0937X2','E0937X3','E0937X9','E0939','E094',
                'E0940','E0941','E0942','E0943','E0944','E0949','E095','E0951','E0952','E0959','E096','E0961','E09610','E09618','E0962',
                'E09620','E09621','E09622','E09628','E0963','E09630','E09638','E0964','E09641','E09649','E0965','E0969','E098','E099',
                'E10','E1010','E1011','E102','E1021','E1022','E1029','E103','E1031','E10311','E10319','E1032','E10321','E103211',
                'E103212','E103213','E103219','E10329','E103291','E103292','E103293','E103299','E1033','E10331','E103311','E103312',
                'E103313','E103319','E10339','E103391','E103392','E103393','E103399','E1034','E10341','E103411','E103412','E103413',
                'E103419','E10349','E103491','E103492','E103493','E103499','E1035','E10351','E103511','E103512','E103513','E103519',
                'E10352','E103521','E103522','E103523','E103529','E10353','E103531','E103532','E103533','E103539','E10354','E103541',
                'E103542','E103543','E103549','E10355','E103551','E103552','E103553','E103559','E10359','E103591','E103592','E103593',
                'E103599','E1036','E1037','E1037X1','E1037X2','E1037X3','E1037X9','E1039','E104','E1040','E1041','E1042','E1043','E1044',
                'E1049','E105','E1051','E1052','E1059','E106','E1061','E10610','E10618','E1062','E10620','E10621','E10622','E10628',
                'E1063','E10630','E10638','E1064','E10641','E10649','E1065','E1069','E108','E109','E11','E110','E1100','E1101','E111',
                'E1110','E1111','E112','E1121','E1122','E1129','E113','E1131','E11311','E11319','E1132','E11321','E113211','E113212',
                'E113213','E113219','E11329','E113291','E113292','E113293','E113299','E1133','E11331','E113311','E113312','E113313',
                'E113319','E11339','E113391','E113392','E113393','E113399','E1134','E11341','E113411','E113412','E113413','E113419',
                'E11349','E113491','E113492','E113493','E113499','E1135','E11351','E113511','E113512','E113513','E113519','E11352',
                'E113521','E113522','E113523','E113529','E11353','E113531','E113532','E113533','E113539','E11354','E113541','E113542',
                'E113543','E113549','E11355','E113551','E113552','E113553','E113559','E11359','E113591','E113592','E113593','E113599',
                'E1136','E1137','E1137X1','E1137X2','E1137X3','E1137X9','E1139','E114','E1140','E1141','E1142','E1143','E1144','E1149',
                'E115','E1151','E1152','E1159','E116','E1161','E11610','E11618','E1162','E11620','E11621','E11622','E11628','E1163',
                'E11630','E11638','E1164','E11641','E11649','E1165','E1169','E118','E119','E13','E130','E1300','E1301','E131','E1310',
                'E1311','E132','E1321','E1322','E1329','E133','E1331','E13311','E13319','E1332','E13321','E133211','E133212','E133213',
                'E133219','E13329','E133291','E133292','E133293','E133299','E1333','E13331','E133311','E133312','E133313','E133319',
                'E13339','E133391','E133392','E133393','E133399','E1334','E13341','E133411','E133412','E133413','E133419','E13349',
                'E133491','E133492','E133493','E133499','E1335','E13351','E133511','E133512','E133513','E133519','E13352','E133521',
                'E133522','E133523','E133529','E13353','E133531','E133532','E133533','E133539','E13354','E133541','E133542','E133543',
                'E133549','E13355','E133551','E133552','E133553','E133559','E13359','E133591','E133592','E133593','E133599','E1336',
                'E1337','E1337X1','E1337X2','E1337X3','E1337X9','E1339','E134','E1340','E1341','E1342','E1343','E1344','E1349','E135',
                'E1351','E1352','E1359','E136','E1361','E13610','E13618','E1362','E13620','E13621','E13622','E13628','E1363','E13630',
                'E13638','E1364','E13641','E13649','E1365','E1369','E138','E139','N251','O240','O2401','O24011','O24012','O24013','O24019',
                'O2402','O2403','O241','O2411','O24111','O24112','O24113','O24119','O2412','O2413','O243','O2431','O24311','O24312','O24313',
                'O24319','O2432','O2433','O244','O2441','O24410','O24414','O24415','O24419','O2442','O24420','O24424','O24425','O24429',
                'O2443','O24430','O24434','O24435','O24439','O248','O2481','O24811','O24812','O24813','O24819','O2482','O2483','O249','O2491',
                'O24911','O24912','O24913','O24919','O2492','O2493','P700','P702','R7303','Z131','Z833','Z8632'
                ]
    }

comorbidity_list = [diag_cd for diag_codes in list(comorbidity_dict.values()) for diag_cd in diag_codes]

#-------------------------------------------------------------------------------------------------------
df_comorbidities = {}

for key, values in comorbidity_dict.items():
    for value in values:
        df_comorbidities[value] = key

# convert to df
comorbid_code_df = pd.DataFrame.from_dict(df_comorbidities , orient = 'index' , columns = ['comorbidity']).reset_index().rename(columns = {'index':'icd_code'})

# list of comorbidities
comorbidity_codes_list = "('"+"','".join(list(comorbidity_list))+"')"
print(comorbidity_codes_list)

query = f'''
select distinct subject_id , icd_code from mimiciv_hosp.diagnoses_icd where icd_code in {comorbidity_codes_list}
'''

comorbidity = pd.read_sql_query(query , engine)

# remove trailing whitespace
comorbidity['icd_code'] = comorbidity['icd_code'].str.rstrip()

# merge
comorbidity_2 = pd.merge(comorbidity , comorbid_code_df , on = 'icd_code')

# add helper
comorbidity_2['helper'] = 1

# test
assert comorbidity_2.pivot_table(index = 'subject_id' , columns = 'comorbidity' , values = 'helper' , aggfunc = 'sum').fillna(0).sum().sum() == len(comorbidity_2) , 'pivot counts not matching'

# pivot
comorbidity_pivot = comorbidity_2.pivot_table(index = 'subject_id' , columns = 'comorbidity' , values = 'helper').fillna(0).reset_index()

# Joining Patients with presence in all tables_____________________________________________________________________________________________________________
charttime_pats = list(set(bloodwork['subject_id']).\
intersection(set(vitalsign['subject_id'])).\
intersection(set(medication_pivot['subject_id'])).\
intersection(set(omr_pivot['subject_id'])).\
intersection(set(race['subject_id']))
)
len(charttime_pats)

df_1 = pd.merge(
    race , bloodwork , 
    on = 'subject_id' , 
    how = 'inner'
)

df_2 = pd.merge(
    df_1 , vitalsign , 
    on = 'subject_id' , 
    how = 'inner'
)

df_3 = pd.merge(
    df_2 , medication_pivot , 
    on = 'subject_id' , 
    how = 'inner'
)

df_4 = pd.merge(
    df_3 , omr_pivot ,
    on = 'subject_id' ,
    how = 'inner'
)

df_5 = pd.merge(
    df_4 , comorbidity_pivot , 
    on = 'subject_id' , 
    how = 'left' # do not want to inner join as not all HF pats will have at least 1 comorbidity
)

print(f'patients {df_5.subject_id.nunique()}')

# Retain all patients_____________________________________________________________________________________________________________
# Time of survival -> from last ICU admission to deathtime / OR first ICU admission?
# Uncensored patients (those that experienced the event, i.e. died)
# Find latest ICU admission
tos_hf = all_hf_pats_0.groupby('subject_id').agg({'admittime':'min' , 'dod':'max' , 'dischtime':'max'})

# add gender and age
tos_hf = pd.merge(
    tos_hf,  all_hf_pats_0[['subject_id','gender','anchor_age']].drop_duplicates(),
    on = 'subject_id',
    how = 'inner'
)

# who are the uncensored patients?
event_pats = tos_hf[tos_hf['dod'].notnull()].copy()
event_pats['time_to_event'] = event_pats['dod'] - event_pats['admittime']
event_pats['time_to_event'] = event_pats['time_to_event'] / np.timedelta64(1, 'D') # convert to hours
event_pats['death'] = 1

# who are the censored patients?
non_event_pats = tos_hf[tos_hf['dod'].isnull()].copy()# For censored patients, their last follow-up was essentially their discharge time
non_event_pats['time_to_event'] = non_event_pats['dischtime'] - non_event_pats['admittime']
non_event_pats['time_to_event'] = non_event_pats['time_to_event'] / np.timedelta64(1, 'D') # convert to hours
# non_event_pats['time_to_event'] = non_event_pats['time_to_event'] + 365 # is this necessary?
non_event_pats['death'] = 0
# non_event_pats.head(3)

pats = pd.concat([
    non_event_pats[['subject_id', 'gender', 'anchor_age', 'time_to_event' , 'death']] , 
    event_pats[['subject_id','gender', 'anchor_age', 'time_to_event' , 'death']]
] , axis = 0)

pats.sort_values('subject_id' , inplace = True)

pats['time_to_event'] = pats['time_to_event'].apply(lambda x: x+1 if x==0 else x)
# QC
print(pats.count())

# QC
assert len(pats) == all_hf_pats_0["subject_id"].nunique() , 'is not patient level table'

# Visualize_____________________________________________________________________________________________________________
grouped = pats.groupby('death')
labels = ['censored' , 'uncensored']
colors = ['slategray' , 'orange']
# Plot grouped histogram
fig, axes = plt.subplots(2 , figsize = (7,5))
            
for category, group in grouped:
    axes[category].hist(group['time_to_event'], bins=50 , density = True , label = labels[category] , color = colors[category])
    axes[category].legend()
    axes[category].set_ylabel('Density')

plt.xlabel('Time to Event')
# plt.title('Cohort-wise Time-to-Event Distribution')

# Save the plot as an PDF file
plt.savefig('cohort_hist.pdf', format='pdf')

plt.show()

# Consolidate_____________________________________________________________________________________________________________
consolidated_pat_tbl = pd.merge(df_5 , pats , on = 'subject_id' , how = 'inner')

# QC
assert len(consolidated_pat_tbl) == consolidated_pat_tbl.subject_id.nunique() , 'not patient level table'

# Store data_____________________________________________________________________________________________________________

# Check if the folder exists
if not os.path.exists(folder_path):
    # Create the folder
    os.makedirs(folder_path)

# Pickle the DataFrame
with open(folder_path+'consolidated_pat_tbl.pickle', 'wb') as file:
    pickle.dump(consolidated_pat_tbl, file)

# QC
assert pd.get_dummies(consolidated_pat_tbl['race']).sum().sum() == consolidated_pat_tbl['subject_id'].nunique()

# Cast as integer
for col in consolidated_pat_tbl.columns:
    if (col != 'race') and (col != 'gender'):
        consolidated_pat_tbl[col] = consolidated_pat_tbl[col].astype('float64')

# OneHotEncode race (ethnicity)!
race_encoder = OneHotEncoder(sparse_output=False)
race_enc = race_encoder.fit_transform(consolidated_pat_tbl[['race']])
mod_df = consolidated_pat_tbl.drop('race' , axis = 1)
mod_df = pd.concat([mod_df , pd.DataFrame(
    race_enc , 
    columns = consolidated_pat_tbl[['race']].drop_duplicates().sort_values('race').values.T[0]
    )
] , axis = 1)

# Leave out one of the one-hot encoded columns so as to not raise multicollinearity issues
mod_df.drop('OTHER' , axis = 1 , inplace = True)

# OneHotEncode gender (sex)
gender_encoder = OneHotEncoder(sparse_output=False)
gender_enc = gender_encoder.fit_transform(mod_df[['gender']])
mod_df = mod_df.drop('gender' , axis = 1)
mod_df = pd.concat([mod_df , pd.DataFrame(
    gender_enc , 
    columns = consolidated_pat_tbl[['gender']].drop_duplicates().sort_values('gender').values.T[0]
    )
] , axis = 1)

# Leave out one of the one-hot encoded columns so as to not raise multicollinearity issues
mod_df.drop('F' , axis = 1 , inplace = True)

# Train Test Split_____________________________________________________________________________________________________________
def train_test_splitter_tv(df , test_size = 0.2 , val_size = 0.2 , patient_col = 'subject_id'):
    # get all patients
    pats = df[patient_col].unique()
    
    # inplace shuffle - uncomment this if you want different cohorts each time
    np.random.shuffle(pats)

    # get splits
    test_pats = pats[:int(test_size*len(pats))]
    val_pats = pats[int(test_size*len(pats)):int(test_size*len(pats))+int(val_size*len(pats))]
    train_pats = pats[int(test_size*len(pats))+int(val_size*len(pats)):]

    # allocate
    df_test = df[df['subject_id'].isin(test_pats)]
    df_val = df[df['subject_id'].isin(val_pats)]
    df_train = df[df['subject_id'].isin(train_pats)]
    
    # check
    assert df_train.subject_id.nunique() + df_test.subject_id.nunique() + df_val.subject_id.nunique() == df.subject_id.nunique()
    return df_train , df_test , df_val

def _preprocess_stuff(df):
    _columns = df.columns.astype('str')
    df.columns = _columns

    # Imputation
    imputer = SimpleImputer(strategy = 'most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df) , columns = _columns)

    # Check for null
    assert np.round(df.notnull().sum()/len(df)).sum() == df.shape[1]


    return df

# train test split
df_train , df_test , df_val = train_test_splitter_tv(mod_df)

# use on train test val data
df_train = _preprocess_stuff(df_train)
df_test = _preprocess_stuff(df_test)
df_val = _preprocess_stuff(df_val)

# Scaling
scaler = StandardScaler()
scaled_columns = [
    'cholesterol', 'sodium', 'hemoglobin','lymphocyte',
    'temperature', 'heartrate', 'resprate', 
    'o2sat', 'sbp', 'dbp','anchor_age','BMI (kg/m2)',
    'Height (Inches)', 'Weight (Lbs)'
]

unscaled_columns = ['acebutolol', 'amlodipine', 'atenolol', 'benazepril', 
                    'candesartan','captopril', 'diltiazem', 'felodipine', 
                    'irbesartan', 'lisinopril','moexipril', 'nadolol', 
                    'nebivolol', 'nicardipine', 'nifedipine','olmesartan', 
                    'propranolol', 'quinapril', 'ramipril', 'telmisartan',
                    'trandolapril', 'valsartan', 'verapamil', 'diabetes', 
                    'hypertension','obesity','ASIAN', 'BLACK', 'HISPANIC', 
                    'NATIVE','WHITE' , 'M', 'time_to_event','death'
]

assert len(scaled_columns) + len(unscaled_columns) ==  len(mod_df.columns) - 1 , f'{len(scaled_columns) + len(unscaled_columns)} and {len(mod_df.columns)} columns not matching'

scale = [([col], StandardScaler()) for col in scaled_columns]
no_scale = [(col, None) for col in unscaled_columns]

x_mapper = DataFrameMapper(scale + no_scale)

# scale train test val data
x_train = pd.DataFrame(x_mapper.fit_transform(df_train).astype('float32') , columns = scaled_columns + unscaled_columns)
x_val = pd.DataFrame(x_mapper.transform(df_val).astype('float32') , columns = scaled_columns + unscaled_columns)
x_test = pd.DataFrame(x_mapper.transform(df_test).astype('float32') , columns = scaled_columns + unscaled_columns)

# Remove 0-var columns (in train) from all 3 sets______________________________________________________________________
def check_var(df):
    zero_var_cols = []
    # Check which columns have zero variance
    for _col in df.columns[:-2]:
        var = df[_col].var()
        if var == 0:
            zero_var_cols.append(_col)
    return zero_var_cols

zero_var_cols = check_var(x_train)

print(f'zero var cols {zero_var_cols}')

# Drop the cols
x_train.drop(zero_var_cols, axis = 1, inplace = True)
x_test.drop(zero_var_cols, axis = 1, inplace = True)
x_val.drop(zero_var_cols, axis = 1, inplace = True)

# Pickle the DataFrame
with open(folder_path+'x_train.pickle', 'wb') as file:
    pickle.dump(x_train, file)

with open(folder_path+'x_val.pickle', 'wb') as file:
    pickle.dump(x_val, file)

with open(folder_path+'x_test.pickle', 'wb') as file:
    pickle.dump(x_test, file)