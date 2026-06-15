# VMan3 Machine Learning Model
This is VMan3 Machine Learning model for predicting CoD. The training and prediction flow now auto-detects the WHO VA instrument version and routes to the matching model artifact.

## Train
```
% python3 train.py \
    --input data/va_2016_tz.csv data/va_2022_ng.csv data/va_2022_es.csv \
    --verbose

```
### Add more options
```
% python3 train.py \
--input data/va_2016_tz.csv data/va_2022_ng.csv data/va_2022_es.csv \
--verbose \
--min_vc 50

````
### Note
Target column should be labelled `pcva_ucod`. The ICD consistency field is `pcva_ucod_icd`.

## Predict
```
% python3 predict.py --model 'models/ccva_model.pkl' --input va_test.csv --output results.csv --verbose
```

Disable DK-based OOD entirely (set to 100% = nothing fails this)
```
python3 predict.py --model models/ --input va_test2.csv --dk-threshold 1.0
```

# Also relax confidence threshold on the old model
```
python3 predict.py --model models/ --input va_test2.csv --dk-threshold 1.0 --ood_threshold 0.05
```