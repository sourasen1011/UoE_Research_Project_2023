This directory just contains two copies of code from https://github.com/havakv/pycox/tree/0e9d6f9a1eff88a355ead11f0aa68bfb94647bf8/pycox/evaluation

```concordance.py -> concordance_copy.py```
```eval_surv.py -> eval_surv_copy.py```

I needed to copy them because the evaluation offered by the original package was not working on my machine due to a package version problem(?) with ```numba```. I just changed that part and now it works.