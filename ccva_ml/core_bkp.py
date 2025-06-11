import sys
import numpy as np 
import pandas as pd
import argparse
import chardet
import matplotlib.pyplot as plt
from time import time

from sklearn.model_selection import ( ShuffleSplit, train_test_split, StratifiedKFold, StratifiedShuffleSplit, KFold )
from sklearn.model_selection import ( RandomizedSearchCV, GridSearchCV, cross_val_score)
from sklearn.preprocessing import (OneHotEncoder,LabelBinarizer,LabelEncoder,StandardScaler,MinMaxScaler,MaxAbsScaler, MultiLabelBinarizer)
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score, precision_recall_fscore_support)

# from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from collections import Counter 

from imblearn.under_sampling import RandomUnderSampler

from xgboost import XGBClassifier

from vman3_dq import change_null_toskipped


def count_na_values(dataframe):
    """
        A function to count the number of NA in dataframe
    """
    return dataframe.isna().sum()   # Return the count of NA values in each column of the DataFrame

def drop_na_columns(dataframe:pd.DataFrame, th:float=0.7):
    """
        A function to drop NA
    """
    return dataframe.dropna(thresh = 0.7*len(dataframe), axis=1)  # Drop columns with more than 70% NA values


def labelEncoder(X):
    encoded_df = pd.DataFrame()
    encoder = LabelEncoder()
    for i in X.columns:
        encoded = encoder.fit_transform(X[i])
        encoded = pd.DataFrame(data=encoded, columns=[i])
        encoded_df = pd.concat([encoded_df, encoded], axis=1)
    return encoded_df
 
def ccva_train(df:pd.DataFrame, rv_min_vc:int=100, drop_na_th:float=0.7, verbose:bool=False):
    """
    Train a Machine Learning model for CCVA

    Parameters
    - df : Dataframe containing training dataset
    - verbose : Whether to print debugging information
    -- rv_min_vc : Response variable minimum value counts: Default is 130
    -- drop_na_th: Drop NA threshold value, Default is 70%

    Returns: 
    - None
    """

    if verbose:
        print("Starting CCVA ML Training Module")

    # Check required columns
    required_columns = ['id10019', 'ageInYears', 'age_group','isNeonatal', 'isChild', 'isAdult']
    #df.columns = df.columns.str.lower().str.replace(" ", "_") # change all columns to lower case

    missing_columns = [
        col for col in required_columns
        if col.lower() not in df.columns.str.lower()
    ]
    if missing_columns:
        print(f"Error: Key important columns are missing for the training dataset: {', '.join(missing_columns)}")
        sys.exit(1)  # Exit with non-zero status


    # Create key columns
    if verbose:
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        # print(f"Number of NULLs before cleaning {df.isna().sum().sum():,}")   
    
    df = change_null_toskipped(df,verbose=verbose)
    
    if verbose:
        print(f"Number of NULLs after  cleaning {df.isna().sum().sum():,}") 

    # drop all colums with sufix _check. These do not provide any relevant informaton
    if verbose:
        print("Creating the training dataset")
    df = df.drop(columns=[col for col in df.columns if "_check" in col])

    # change everything to lower case
    df.columns = df.columns.str.lower().str.replace(" ", "_") # change all columns to lower case

    # # Prepare ML dataset
    # # 1. Demographic
    columns_to_select = ['Id10019', 'ageInYears', 'age_group','isNeonatal', 'isChild', 'isAdult']
    columns_to_select = [col.lower() for col in columns_to_select]
    missing_columns = [col for col in columns_to_select if col not in df.columns]

    if missing_columns:
        print("The following column(s) do not exist in the DataFrame:", missing_columns)
        print("\n Cannot proceed without this/these column")
        sys.exit(1)
    else:
        df_demographic = df.loc[:, columns_to_select]

    # 2. Accident and Injuries
    columns_to_select = ['id10077', 'id10079', 'id10080', 'id10081', 'id10082', 'id10083', 
    'id10084', 'id10085', 'id10086', 'id10087', 'id10088', 'id10089', 'id10090', 'id10091', 
    'id10092', 'id10093', 'id10094', 'id10095', 'id10096', 'id10097', 'id10098', 'id10099', 
    'id10100']
    columns_to_select = [col.lower() for col in columns_to_select]
    missing_columns = [col for col in columns_to_select if col not in df.columns]

    # If there are missing columns, print an error message
    if missing_columns:
        print("The following column(s) do not exist in the DataFrame:", missing_columns)
        print("\n Cannot proceed without this/these column")
        sys.exit(1)
    else:
        # Select the columns from the DataFrame
        df_accdnt_inj = df.loc[:, columns_to_select]
    
    # 3.Medical history associated with final illness
    columns_to_select = ['id10125', 'id10126', 'id10127', 'id10128', 'id10129', 'id10130', 
                         'id10131', 'id10132', 'id10133', 'id10134', 'id10135', 'id10136', 
                         'id10137', 'id10138', 'id10139', 'id10140', 'id10141', 'id10142', 
                         'id10143', 'id10144']    

    columns_to_select = [col.lower() for col in columns_to_select]
    missing_columns = [col for col in columns_to_select if col not in df.columns]

    if missing_columns:
        print("The following column(s) do not exist in the DataFrame:", missing_columns)
        print("\n Cannot proceed without this/these column")
        sys.exit(1)
    else:
        df_med_hist_final_ill = df.loc[:, columns_to_select]

    # 4.General Symptoms
    columns_to_select = ['id10147', 'id10173_a', 'id10173', 'id10174', 'id10181', 'id10186', 
    'id10188', 'id10189', 'id10193', 'id10194', 'id10200', 'id10204', 'id10207', 'id10208', 
    'id10210', 'id10212', 'id10214', 'id10219', 'id10223', 'id10227', 'id10228', 'id10230', 
    'id10233', 'id10237', 'id10238', 'id10241', 'id10243', 'id10244', 'id10245', 'id10246', 
    'id10247', 'id10249', 'id10252', 'id10253', 'id10258', 'id10261', 'id10264', 'id10265', 
    'id10267', 'id10268', 'id10270', 'id10295', 'id10296', 'id10304', 'id10305', 'id10306', 
    'id10307', 'id10310']

    # Check if any of the columns in columns_to_select do not exist in the DataFrame
    columns_to_select = [col.lower() for col in columns_to_select]
    missing_columns = [col for col in columns_to_select if col not in df.columns]
    if missing_columns:
        print("The following column(s) do not exist in the DataFrame:", missing_columns)
        print("\n Cannot proceed without this/these column")
        sys.exit(1)
    else:
        df_general_sympt = df.loc[:, columns_to_select]
        

    # 5. Risk factors
    columns_to_select = ['id10411', 'id10412', 'id10413']
    columns_to_select = [col.lower() for col in columns_to_select]
    missing_columns = [col for col in columns_to_select if col not in df.columns]
    if missing_columns:
        print("The following column(s) do not exist in the DataFrame:", missing_columns)
        print("\n Cannot proceed without this/these column")
        sys.exit(1)
    else:
        df_risk_factors = df.loc[:, columns_to_select]
    
    # 6. Health service utilization
    columns_to_select = ['id10418', 'id10419', 'id10420', 'id10421', 'id10422', 'id10423', 'id10424','id10425', 'id10427', 'id10432']
    columns_to_select = [col.lower() for col in columns_to_select]
    missing_columns = [col for col in columns_to_select if col not in df.columns]

    if missing_columns:
        print("The following column(s) do not exist in the DataFrame:", missing_columns)
        print("\n Cannot proceed without this/these column")
        sys.exit(1)
    else:
        df_health_serv_util = df.loc[:, columns_to_select]

    # 7.Background context
    columns_to_select = ['id10450', 'id10455', 'id10456', 'id10457', 'id10458', 'id10459']
    columns_to_select = [col.lower() for col in columns_to_select]
    missing_columns = [col for col in columns_to_select if col not in df.columns]

    if missing_columns:
        print("The following column(s) do not exist in the DataFrame:", missing_columns)
        print("\n Cannot proceed without this/these column")
        sys.exit(1)
    else:
        df_backgrnd_context = df.loc[:, columns_to_select]
    
    # responde column
    df_response = df.loc[:, 'cod_who_ucod']

    # concat into a datafram
    # concat the filtered dataframes into one dataframe
    df2 = pd.concat(
        [
            df_demographic, 
            df_accdnt_inj, 
            df_med_hist_final_ill, 
            df_general_sympt, 
            df_risk_factors, 
            df_health_serv_util, 
            df_backgrnd_context, 
            df_response
        ],
        axis=1
    )

    print(f"Dimension of the training dataset: {df2.shape}")

    # keep records with sufficient records in the response variable for training purposes. 
    df3 = df2[df2['cod_who_ucod'].isin(df['cod_who_ucod'].value_counts()[lambda x: x > rv_min_vc].index)]
    
    print(f"\nSelected CoD with at least {rv_min_vc} records")
    print(df3.cod_who_ucod.value_counts())

    # Create individual dataframe of the selected CoD
    cod_dfs = {cod: df3[df3['cod_who_ucod'] == cod] for cod in df3['cod_who_ucod'].unique()}

    df4 = []
    for cod in cod_dfs:
        cod_df = drop_na_columns(cod_dfs[cod])  # drop NULL and get the new dataframe
        print(f"Dataframe: {cod}, Shape before dropping NA: {cod_dfs[cod].shape}, Shape after droppoing NA: {cod_df.shape}")
        df4.append(cod_df)

    # concatenate back
    df5 = pd.concat(df4, axis=0)
    print(f"Before dropping NA (Rows): \n {df5['cod_who_ucod'].value_counts()}")

    # Drop NA rows
    df5_noNA = df5.dropna()
    print(f"After droppoing NA (Rows): \n {df5_noNA['cod_who_ucod'].value_counts()}")
    print("Training completed")

    # keep with at least 100 value counts
    training_df = df5_noNA[df5_noNA['cod_who_ucod'].isin(df5_noNA['cod_who_ucod'].value_counts()[lambda x: x > 100].index)]

    print(f"Keep at least with 100 records in the COD (Rows): \n {training_df['cod_who_ucod'].value_counts()}")

    # Prepare training and testing dataset
    # select features and target
    X = training_df.drop('cod_who_ucod', axis=1)
    y = training_df['cod_who_ucod']

    # check if X contains categorical and numerical features
    # check how many columns are categorical and numerical
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(exclude=['object']).columns

    print(f'The number of categorical columns is {len(categorical_columns)}')
    print(f'The number of numerical columns is {len(numerical_columns)}')

    # separate the categorical and numerical columns
    categorical_var = X[[i for i in X.columns if i not in numerical_columns]]

    # encode the categorical columns
    cat_enco = labelEncoder(categorical_var)

    # then concatenate the encoded categorical columns with the numerical columns
    X_encoded = pd.concat(
        [
            X[numerical_columns].reset_index(drop=True),
            cat_enco.reset_index(drop=True),
        ],
        axis=1
    )

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, 
        y, 
        test_size=0.3,
        shuffle=True,
        random_state=42
    )

    print('The shape of X train : {}'.format(X_train.shape))
    print('The shape of y train : {}'.format(y_train.shape))
    print('The shape of X test : {}'.format(X_test.shape))
    print('The shape of y test : {}'.format(y_test.shape))

    # Scale the features using StandardScaler, Mean of 0, SD = 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_folds = 5 # split data into five folds
    seed = np.random.randint(0, 81470) # random seed value
    scoring = 'accuracy' # metric for model evaluation

    # specify cross-validation strategy

    kf = KFold(
                n_splits = num_folds,
                shuffle = True,
                random_state = seed
            )

    # make a list of models to test

    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('LR', LogisticRegression( #multi_class = 'ovr',       # commented this, as it was giving warnings
                                             max_iter = 2000,random_state = seed)))
    models.append(('SVM', SVC( kernel = 'linear',gamma = 'auto',random_state = seed)))
    models.append(('RF', RandomForestClassifier(n_estimators = 500,random_state = seed)))
    models.append(('XGB', XGBClassifier(random_state = seed,n_estimators = 500)))

    # check if the labels are balanced
    print(y_train.value_counts())

    # balance the labels using RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train_scaled, y_train)
    print('The number of samples after resampling:', Counter(y_train_resampled))

    # Evaluate models to get the best performing model
    label_encoder = LabelEncoder()
    y_train_resampled_encoded = label_encoder.fit_transform(y_train_resampled)

    results = []
    names = []

    for name, model in models:
        cv_results = cross_val_score(
            model,
            np.asarray(X_train_resampled),
            #np.asarray(y_train_resampled),
            np.asarray(y_train_resampled_encoded),
            cv=kf,
            scoring=scoring
        )
        results.append(cv_results)
        names.append(name)
        msg = f'Cross validation score for {name}: {cv_results.mean():.2%} ± {cv_results.std():.2%}'
        print(msg)
    
    # big LOOP
    # TUNNING THE SELECTED MODEL

    # Set validation procedure
    num_folds = 5 # split training set into 5 parts for validation
    num_rounds = 5 # increase this to 5 or 10 once code is bug-free
    # seed = 4 # pick any integer. This ensures reproducibility of the tests
    scoring = 'accuracy' # score model accuracy

    # prepare matrices of results
    kf_results = pd.DataFrame() # model parameters and global accuracy score
    kf_per_class_results = [] # per class accuracy scores
    save_predicted, save_true = [], [] # save predicted and true values for each loop

    start = time()

    # Specify model

    classifier = RandomForestClassifier()

    # Optimizing hyper-parameters for random forest

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    print(random_grid)


    for round in range(num_rounds):
        
        # cross validation and splitting of the validation set
        for train_index, test_index in kf.split(np.asarray(X_train_resampled), np.asarray(y_train_resampled)):
            X_train_set, X_val = np.asarray(X_train_resampled[train_index]), np.asarray(X_train_resampled[test_index])
            y_train_set, y_val = np.asarray(y_train_resampled[train_index]), np.asarray(y_train_resampled[test_index])

            print('The shape of X train set : {}'.format(X_train_set.shape))
            print('The shape of y train set : {}'.format(y_train_set.shape))
            print('The shape of X val : {}'.format(X_val.shape))
            print('The shape of y val : {}'.format(y_val.shape))

            # generate models using all combinations of settings

            # RANDOMSED GRID SEARCH
            # Random search of parameters, using 5 fold cross validation, 
            # search across 100 different combinations, and use all available cores

            n_iter_search = 10
            rsCV = RandomizedSearchCV(
                                        verbose=1,
                                        estimator=classifier, 
                                        param_distributions=random_grid, 
                                        n_iter=n_iter_search, 
                                        scoring=scoring, 
                                        cv=kf, 
                                        refit=True, 
                                        n_jobs=-1
                                    )
            
            rsCV_result = rsCV.fit(X_train_set, y_train_set)

            # print out results and give hyperparameter settings for best one
            means = rsCV_result.cv_results_['mean_test_score']
            stds = rsCV_result.cv_results_['std_test_score']
            params = rsCV_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%.2f (%.2f) with: %r" % (mean, stdev, param))

            # print best parameter settings
            print("Best: %.2f using %s" % (rsCV_result.best_score_,
                                        rsCV_result.best_params_))

            # Insert the best parameters identified by randomized grid search into the base classifier
            best_classifier = classifier.set_params(**rsCV_result.best_params_)
        
        
            best_classifier.fit(X_train_set, y_train_set)

            # predict test instances 

            y_pred = best_classifier.predict(X_val)
            # y_test = np.delete(y_res, train_index, axis=0)
            local_cm = confusion_matrix(y_val, y_pred)
            local_report = classification_report(y_val, y_pred)

            # zip predictions for all rounds for plotting averaged confusion matrix
            
            for predicted, true in zip(y_pred, y_val):
                save_predicted.append(predicted)
                save_true.append(true)

        #    # append feauture importances
        #     local_feat_impces = pd.DataFrame(
        #                                         best_classifier.feature_importances_,
        #                                         index=features.columns
        #                                     ).sort_values(by=0, ascending=False)
        
            # summarizing results
            local_kf_results = pd.DataFrame(
                                                [
                                                    ("Accuracy", accuracy_score(y_val, y_pred)), 
                                                    ("TRAIN", str(train_index)), 
                                                    ("TEST", str(test_index)), 
                                                    ("CM", local_cm), 
                                                    ("Classification report", local_report), 
                                                    ("y_test", y_val),
                                                    # ("Feature importances", local_feat_impces.to_dict())
                                                ]
                                            ).T
            
            local_kf_results.columns = local_kf_results.iloc[0]
            local_kf_results = local_kf_results[1:]
            # kf_results = kf_results.append(local_kf_results)

            kf_results = pd.concat(
                                    [kf_results, local_kf_results],
                                    axis=0,
                                    join='outer'
                                ).reset_index(drop=True)

            # per class accuracy
            local_support = precision_recall_fscore_support(y_val, y_pred)[3]
            local_acc = np.diag(local_cm) / local_support
            kf_per_class_results.append(local_acc)

    elapsed = time() - start
    print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(elapsed / 60, elapsed))

    # Predict test/unseen data
    # scale the test data
    X_test_scl = scaler.transform(X = X_test)

    # Predict test set
    predictions = best_classifier.predict(X_test_scl)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Summarising precision, f_score, and recall for the unseen test set
    cr = classification_report(y_test, predictions)
    print('Classification report : {}')
    print(cr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--verbose", type=bool, required=False, help="Print output to terminal")
    args = parser.parse_args()
    
    with open(args.input, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']

    # Read file with detected encoding
    print("\n Reading the input file")
    df = pd.read_csv(args.input,encoding = encoding,low_memory = False)
    ccva_train(df, verbose=args.verbose)