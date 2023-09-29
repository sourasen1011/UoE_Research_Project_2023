# **UoE_Research_Project_2023**

Repository to store all work done on M.Sc. Data Science dissertation. For the paper, go to https://github.com/sourasen1011/UoE_Research_Project_2023/blob/dev/presentations/my_thesis/01/thesis.pdf.

The full data needs a DUA to be signed and a course to be completed, however, here is a demo of the data - https://physionet.org/content/mimic-iv-demo/2.2/

### Step By Step Run(s)
Once you have unzipped the contents, open up a terminal.
Navigate to your current folder. Set up a virtual environment.
```
python -m venv my_virtual_env
```
Activate the environment.
```
my_virtual_env\Scripts\activate
```
Install the dependencies from the requirements file.
```
pip install -r requirements.txt
```
First, set up the data by running the preprocessing files.
```
cd preprocessing
python time_invariant_preprocessing.py
python time_invariant_preprocessing.py
```
This will create a ```data``` folder storing all the required preprocessed files. Then you can independently run any of the ```.ipynb``` files in the ```examples``` directory. Make sure to use the same virtual environment to run these files.

For experiments, you can run the following from the terminal.
```
cd ..\experiments
python repeat_fits.py --iter 5 --verbose True
```
This will print out a dictionary of evaluation metrics to the terminal. Copy that into the ```experiment_fits.ipynb``` file to visualize them. Otherwise, the ```experiments.ipynb``` file has other independent experiments as well.S

#### ```preprocessing```

The ```preprocessing``` directory contains two .py files for cleaning and ingesting the data from MIMIC-IV. This site goes into detail about the dataset - https://physionet.org/content/mimiciv/2.2/. This github repository talks about setting up the data locally - https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv.


#### ```data```

Once the data is processed, they are stored in the ```data``` directory. This directory does not need to be made beforehand, however, it does need to be named ```data```.

#### ```nn_survival_analysis```

The main modules are contained within ```nn_survival_analysis```. The two proposed models - Time-Invariant and Time-Variant have their own .py files within this directory. Other traditional models as well as neural network models also have their own .py files here. The ```nn_survival_analysis``` directory also contains a ```config.json``` file, which contains all the hyperparameters for training, testing and evaluating.


#### ```evals```

The ```evals``` directory contains two required .py files for evaluation, one of which is called directly in both the ```time_invariant_surv.py``` and ```time_variant_surv.py``` files in the evaluation functions for their respective classes. 

Traditional models - Cox Proportional Hazards, Weibull Accelerated Failure Time model and Random Survival Forest are present in the ```traditional_models.py``` file. Deep Survival Machines and PyCox (with Logisitc Hazards) are present in the ```other_nn_models.py``` file. The two proposed models are present in the ```models.py``` file.


#### ```experiments```

The ```experiments``` directory contains experiments done with all the fitters on MIMIC-IV data. The ```repeat_fits.py``` file essentially calls the ```nn_survival_analysis.run_fitters.py``` file ```iter``` number of times. This allows the generation of distributions for the c-index and the IBS.

There are some bugs with this however. Sometimes, the RSF fitter returns the exact same value over multiple fits when called in conjunction with other fitters, but not when called alone (by commenting out every other fitter in the ```nn_survival_analysis.run_fitters.py``` file). The ```experiment_fits.ipynb``` file tests out some extensions like adding PCA and k-means clustering before feeding the data to the Time-Invariant Survival (TIS) model. The k-means functionality is included within the TIS model, but is kept to 1 cluster, as increasing that number causes the performance to **degrade**. Time-Variant Survival (TVS) is not equipped with clustering. This ```experiments_fits.ipynb``` file also produces graphs for the distributions of the scores.

#### ```examples```

The ```examples``` directory contains experiments with some other datasets as well as MIMIC-IV.
1. FLCHAIN
FLCHAIN: contains half of the data collected during a study about the possible relationship between serum FLC and mortality.
https://rdrr.io/cran/survival/man/flchain.html

2. METABRIC
The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) uses gene and protein expression profiles to determine new breast cancer subgroups in order to help physicians provide better treatment recommendations.
https://ega-archive.org/studies/EGAS00000000083

3. SUPPORT
The Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT) is a larger study that researches the survival time of seriously ill hospitalized adults. The dataset consists of 9,105 patients and 14 features for which almost all patients have observed entries (age, sex, race, number of comorbidities, presence of diabetes, presence of dementia, presence of cancer, mean arterial blood pressure, heart rate, respiration rate, temperature, white blood cell count, serum’s sodium, and serum’s creatinine).
https://pubmed.ncbi.nlm.nih.gov/9610025/

The consensus among the above three datasets is that tree-based models with default parameters generally outperform all other models. It will likely take a lot of hand-tuning to get TIS or PyCox to perform as well.

#### ```resources```

Although the SUPPORT and METABRIC datasets are sourced from the ```pycox``` module, the FLCHAIN data is present in the ```resources/other_data``` folder.