This directory contains some experiments.

The repeat_fits.py file essentially calls the ```nn_survival_analysis.run_fitters.py``` file ```iter``` number of times. This allows the generation of distributions for the c-index and the IBS. This would be the way to do it via the terminal. You can then use the output dictionary to visualize the distributions of the scores via the ```experiment_fits.ipynb``` file.

```
Set-ExecutionPolicy Unrestricted -Scope Process
{project_env}\Scripts\activate
cd experiments
python repeat_fits.py --iter 2 --verbose True
```

There are some bugs with this however. Sometimes, the RSF fitter returns the exact same value over multiple fits when called in conjunction with other fitters, but not when called alone (by commenting out every other fitter in the ```nn_survival_analysis.run_fitters.py``` file)

The ```experiment_fits.ipynb``` file tests out some extensions like adding PCA and k-means clustering before feeding the data to the Time-Invariant Survival (TIS) model. The k-means functionality is included within the TIS model, but is kept to 1 cluster, as increasing that number causes the performance to degrade. Time-Variant Survival (TVS) is not equipped with clustering.