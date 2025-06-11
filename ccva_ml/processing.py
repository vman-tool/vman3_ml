# vman3_ml/vman3_ml/processing.py
import pandas as pd
import numpy as np
from vman3_dq import change_null_toskipped
from sklearn.preprocessing import StandardScaler, LabelEncoder
import chardet

class DataPreprocessor:
    def __init__(self, verbose=False, min_vc=130, na_threshold=0.7):
        self.verbose = verbose
        self.min_vc = min_vc
        self.na_threshold = na_threshold
        self.final_training_columns = None  # To store the final columns used in training
        
    def load_data(self, file_path):
        """Load and preprocess raw data"""
        with open(file_path, 'rb') as file:
            encoding = chardet.detect(file.read())['encoding']
        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
        #return self._preprocess_data(df)
        return df
    
    def _preprocess_data(self, df):
        """Internal preprocessing steps"""
        if self.verbose:
            print("Starting data preprocessing")
        
        # Drop unnecessary columns
        df = df.drop(columns=[col for col in df.columns if "_check" in col])

        # Clean data using vman3_dq
        df = change_null_toskipped(df, verbose=self.verbose)

        # Standardize column names
        df.columns = (
            df.columns
            .str.strip()  # First remove any whitespace
            .str.lower()  # Convert to lowercase
            .str.replace(r'[\s/]+', '_', regex=True)  # Replace spaces and slashes with underscore
            .str.replace(r'_+', '_', regex=True)  # Replace multiple underscores with single
            .str.replace(r'[^\w_]', '', regex=True)  # Remove any other special chars
            .str.strip('_')  # Remove leading/trailing underscores
        )
       
        clean_df = self._merge_duplicate_columns(df)
        return clean_df
    
    def _prepare_training_data(self, df, target_col='cod_who_ucod'):
        """Prepare features and optionally target for training/prediction
        
        Args:
            df: Input DataFrame
            min_vc: Minimum value count for target categories
            na_threshold: Threshold for dropping columns with too many NAs
            target_col: Name of target column (None for prediction mode)
        
        Returns:
            Tuple of (features, target) or (features, None) in prediction mode
        """
        # Feature selection and validation
        feature_sets = self._get_feature_sets()
        dfs = []

        for name, columns in feature_sets.items():
            missing = [col for col in columns if col.lower() not in [c.lower() for c in df.columns]]
            if missing:
                raise ValueError(f"Missing important column {name}: {missing}")
            dfs.append(df[columns])

        # Combine features and optionally target
        if target_col is not None:
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
            full_df = pd.concat(dfs + [df[target_col]], axis=1)
        else:
            full_df = pd.concat(dfs, axis=1)

        # Apply filters only if we have a target column
        if target_col is not None:
            clean_df = self._droprows_by_value_counts(full_df, target_col, self.min_vc)
            clean_df = self._dropcols_by_threshold(clean_df, self.na_threshold)
            
            if self.verbose:
                print(f"Before dropping NA\n{clean_df[target_col].value_counts()}") 
        else:
            clean_df = full_df.copy()

        # Handle missing values
        for col in clean_df.columns:
            if pd.api.types.is_string_dtype(clean_df[col]):
                clean_df[col] = clean_df[col].fillna('dk').replace({'':'dk'})
            else:
                clean_df[col] = clean_df[col].fillna(-999)
        
        if target_col is not None and self.verbose:
            print(f"Training dataset contains the following causes\n{clean_df[target_col].value_counts()}") 

        # check for duplicates
        duplicates = clean_df.columns.duplicated()
        if duplicates.any():
                print(f"Warning: Found {duplicates.sum()} duplicates while preparing training data:")
                print(clean_df.columns[duplicates].tolist())
                # Remove duplicates while preserving order
                clean_df = clean_df.loc[:, ~duplicates]


        # Store the final columns used for training/prediction
        self.final_training_columns = list(dict.fromkeys(
            clean_df.drop(target_col, axis=1).columns if target_col is not None 
            else clean_df.columns
        ))
        
        return (
            clean_df.drop(target_col, axis=1), 
            clean_df[target_col] if target_col is not None else None
        )
    
    
        # 'Malaria': ['id10077','id10126','id10127','id10128','id10130','id10131','id10133','id10134','id10135','id10136','id10137',
        #         'id10142','id10143','id10144','id10148','id10149','id10152','id10153','id10159','id10166','id10168','id10173',
        #         'id10174','id10181','id10186','id10188','id10189','id10193','id10194','id10200','id10204','id10207','id10208',
        #         'id10210','id10214','id10219','id10223','id10227','id10230','id10233','id10238','id10241','id10243','id10244',
        #         'id10245','id10246','id10247','id10249','id10252','id10253','id10258','id10261','id10264','id10265','id10267',
        #         'id10268','id10270']

        # 'HIV/AIDS' : ['id10126','id10127','id10128','id10129','id10130','id10131','id10132','id10133','id10134','id10135','id10136',
        #               'id10137','id10138','id10139','id10140','id10141','id10142','id10143','id10144','id10152','id10153','id10159',
        #               'id10166','id10168','id10173','id10174','id10181','id10186','id10188','id10189','id10193','id10194','id10200',
        #               'id10204','id10207','id10208','id10210','id10212','id10214','id10219','id10223','id10227','id10228','id10230',
        #               'id10233','id10237','id10238','id10241','id10243','id10244','id10245','id10246','id10249','id10252','id10253',
        #               'id10258','id10261','id10264','id10265','id10267','id10268','id10270']

    def _get_feature_sets(self):
        """Define feature groupings"""
        return {
            'demographic': ['id10019','id10059','id10063','id10064','id10065','ageinyears', 'age_group', 'isneonatal','ischild', 'isadult'],
            'accident_injuries': ['id10077', 'id10079', 'id10080', 'id10081','id10082', 'id10083', 'id10084', 'id10085', 'id10086', 
                                  'id10087', 'id10088', 'id10089', 'id10090','id10091', 'id10092', 'id10093', 'id10094', 'id10095', 
                                  'id10096', 'id10097', 'id10098', 'id10099','id10100'],
            'medical_history': ['id10123','id10125', 'id10126', 'id10127', 'id10128','id10129', 'id10130', 'id10131', 'id10132', 'id10133',
                                'id10134', 'id10135', 'id10136', 'id10137','id10138', 'id10139', 'id10140', 'id10141', 'id10142',
                                'id10143', 'id10144','id10148','id10149'],
            'general_symptoms': ['id10147', 'id10148_a','id10173_a', 'id10173', 'id10174','id10181', 'id10186', 'id10188', 'id10189', 'id10193',
                                 'id10194', 'id10195','id10200', 'id10204', 'id10207','id10208', 'id10210', 'id10212', 'id10214', 'id10219',
                                 'id10223', 'id10227', 'id10228', 'id10230','id10233', 'id10237', 'id10238', 'id10241', 'id10243',
                                 'id10244', 'id10245', 'id10246','id10247', 'id10249', 'id10252', 'id10253', 'id10258', 'id10261',
                                 'id10264', 'id10265','id10267', 'id10268', 'id10270', 'id10295', 'id10296', 'id10304', 'id10305',
                                 'id10306','id10307', 'id10310','id10152','id10153','id10158','id10159','id10162','id10163','id10161',
                                 'id10166','id10168','id10182','id10183','id10185','id10187','id10191','id10192','id10199','id10201','id10202'],
            'risk_factors': ['id10411', 'id10412', 'id10413','id10414','id10415','id10416'],
            'hs_utilization': ['id10418', 'id10419', 'id10420', 'id10421', 'id10422', 'id10423', 'id10424','id10425','id10426', 'id10427',
                               'id10428','id10429','id10430','id10431','id10432','id10433','id10435','id10437','id10438','id10445','id10446'],
            'background_context':['id10450','id10451','id10452','id10453','id10454', 'id10455', 'id10456', 'id10457', 'id10458', 'id10459']
        }
    
    def _droprows_by_value_counts(self, df, column, threshold):
        """Filter dataframe by value counts in specified column"""
        vc = df[column].value_counts()
        return df[df[column].isin(vc[vc > threshold].index)] 
    

    def _drop_na_columns(self, dataframe:pd.DataFrame, th:float=0.7):
        return dataframe.dropna(thresh = 0.7*len(dataframe), axis=1)  # Drop columns with more than 70% NA values

    def _dropcols_by_threshold(self, df, th:float=0.7):
        cod_dfs = {cod: df[df['cod_who_ucod'] == cod] for cod in df['cod_who_ucod'].unique()}

        temp_df = []
        for cod in cod_dfs:
            cod_na_dropped = self._drop_na_columns(cod_dfs[cod])
            if self.verbose:
                print(f"Dataframe: {cod}, Shape before dropping NA: {cod_dfs[cod].shape}, Shape after dropping NA: {cod_na_dropped.shape}")
            temp_df.append(cod_na_dropped)
        return pd.concat(temp_df, axis=0)
    
    def _encode_features(self, X):
        """Encode categorical features with robust 1D handling"""
        # Store original index to ensure consistent length
        original_index = X.index if hasattr(X, 'index') else pd.RangeIndex(len(X))
        
        # Ensure we're working with a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Make a copy to avoid modifying original data
        X = X.copy()
        
        # Process each categorical column
        encoders = {}
        encoded_dfs = []
        
        for col in X.select_dtypes(include=['object']).columns:
            try:
                # Convert column to 1D Series
                col_data = X[col]
                if isinstance(col_data, pd.DataFrame):  # Handle case where column is 2D
                    col_data = col_data.iloc[:, 0]  # Take first column
                
                # Standardize missing values
                col_data = col_data.astype(str).replace(['nan', '', ' '], 'dk')
                
                # Get unique values safely
                unique_vals = pd.Series(col_data).unique()
                
                # Handle yes/no/dk columns specially
                if set(unique_vals).issubset({'yes', 'no', 'dk'}):
                    mapping = {'yes': 1, 'no': 0, 'dk': -1}
                    encoded = col_data.map(mapping)
                    encoders[col] = mapping
                else:
                    le = LabelEncoder()
                    encoded = le.fit_transform(col_data)
                    encoders[col] = le
                    
                encoded_dfs.append(pd.DataFrame({col: encoded}, index=original_index))
                
            except Exception as e:
                if self.verbose:
                    print(f"Error encoding column {col}: {str(e)}")
                # Fallback to simple numeric encoding
                col_data = X[col].iloc[:, 0] if isinstance(X[col], pd.DataFrame) else X[col]
                encoded = pd.factorize(col_data.astype(str))[0]
                encoders[col] = None
                encoded_dfs.append(pd.DataFrame({col: encoded}, index=original_index))
        
        # Combine features while preserving original index
        numeric_df = X.select_dtypes(exclude=['object'])
        full_df = pd.concat([numeric_df] + encoded_dfs, axis=1)
        
        # Verify we maintained the same number of samples
        if len(full_df) != len(original_index):
            raise ValueError(
                f"Encoding changed number of samples from {len(original_index)} to {len(full_df)}"
            )
        
        return full_df, encoders
    
    def _scale_features(self, X):
        """Scale features while preserving column names"""
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
        
        # Convert to DataFrame if it isn't already
        if isinstance(X, np.ndarray):
            if not hasattr(self, 'feature_names_in_'):
                raise ValueError("Can't scale numpy array without feature names")
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        
        # Scale and return as DataFrame
        X_scaled = self.scaler.fit_transform(X)

        # Check and remove duplicates
        scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        duplicates = scaled_df.columns.duplicated()
        if duplicates.any():
                print(f"Warning: The following columns were found/created and removed while preparing the training dataset: {scaled_df.columns[duplicates].tolist()}")
                scaled_df = scaled_df.loc[:, ~duplicates]

        return scaled_df, self.scaler
    
    def _encode_target(self, y):
        """Encode target variable"""
        if len(y.shape) > 1:
            raise ValueError("Target variable y must be 1-dimensional. Got shape {}".format(y.shape))
    
        le = LabelEncoder()
        try:
            y_encoded = le.fit_transform(y)
            self.classes_ = le.classes_  # Store the original class labels
            self.class_mapping = dict(zip(y_encoded, y))
            return y_encoded, le
        except Exception as e:
            raise ValueError(f"Error encoding target variable: {str(e)}")
        
    def _validate_input_data(self, df):
        """Comprehensive data quality checks before prediction"""
        errors = []
        warnings = []
        
        # 1. Check column presence
        missing_cols = set(self.final_training_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # 2. Check data types
        for col in self.feature_encoders.keys():
            if col in df.columns:
                if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_numeric_dtype(df[col]):
                    warnings.append(f"Column {col} has unexpected dtype: {df[col].dtype}")
        
        # 3. Check for empty/NA columns
        empty_cols = []
        for col in self.feature_encoders.keys():
            if col in df.columns:
                if df[col].isna().all() or df[col].eq('').all():
                    empty_cols.append(col)
        if empty_cols:
            warnings.append(f"Completely empty columns: {empty_cols}")
        
        # 4. Check for problematic values
        problematic_records = {}
        for col in self.feature_encoders.keys():
            if col in df.columns:
                # Check for non-string values in categorical columns
                if col in self.feature_encoders and not pd.api.types.is_string_dtype(df[col]):
                    problematic_records[col] = df[~df[col].astype(str).str.isalnum()].index.tolist()
        
        if problematic_records:
            warnings.append(f"Potential problematic values in columns: {problematic_records}")
        
        # Return or raise errors
        if errors:
            raise ValueError("\n".join(errors))
        
        if warnings and self.verbose:
            print("\nData Quality Warnings:")
            print("\n".join(warnings))
        
        return True

    def _merge_duplicate_columns(self, df):
        """
        Merge duplicate columns in a DataFrame by column name, keeping non-empty content,
        and return a DataFrame with unique column names.
        
        Returns:
            DataFrame with duplicates merged
            Dictionary of merge operations performed
        """
        # Make a copy to avoid changing the original
        df_clean = df.copy()
        duplicates_report = {}
        
        # Find duplicate columns (case-sensitive)
        col_counts = df_clean.columns.value_counts()
        duplicate_cols = col_counts[col_counts > 1].index.tolist()
        
        for col in duplicate_cols:
            # Find all columns with this duplicate name
            matching_cols = [c for c in df_clean.columns if c == col]
            
            if len(matching_cols) > 1:
                # Create merged column (first non-null value across duplicates)
                merged = (
                    df_clean[matching_cols]
                    .astype(str)
                    .replace({'nan': None, '': None})
                    .bfill(axis=1)
                    .iloc[:, 0]
                )
                
                # Record merge operation
                duplicates_report[col] = {
                    'kept': col,
                    'dropped': matching_cols[1:],
                    'action': 'merged with first non-empty value kept'
                }
                
                # Remove all duplicates and add merged column
                df_clean = df_clean.drop(columns=matching_cols)
                df_clean[col] = merged
        
        # Optional verbose reporting
        if self.verbose and duplicates_report:
            print(f"Merged {len(duplicates_report)} duplicate column groups")
            for col, info in duplicates_report.items():
                print(f"- Kept '{info['kept']}', dropped {info['dropped']}")
        
        return df_clean

        
    