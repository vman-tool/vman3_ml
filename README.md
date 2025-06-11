# VMan3 Machine Learning Model
This is VMan3 Machine Learning model for predicting CoD. This model uses the WHO 2016 Questionnaire standard 

## Train
```
% python3 train.py --input '/Users/ilyatuu/Documents/coding/vman3/datascience/va_for_ai.csv' --verbose
```
### Add more options
```
% python3 train.py --input '/Users/ilyatuu/Documents/coding/vman3/datascience/va_for_ai.csv' --verbose --min_vc 200
````
### Note
Target column should be labelled ```cod_who_ucod```

## Predict
```
% python3 predict.py --model 'models/ccva_model.pkl' --input test1.csv --output results.csv --verbose
```