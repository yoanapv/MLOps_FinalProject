from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import pandas as pd
import re
import numpy as np
    
class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to impute missing values in categorical variables.

    Parameters:
        variables (list or str, optional): List of column names (variables) to impute missing values for.
            If a single string is provided, it will be treated as a single variable. Default is None.

    Attributes:
        variables (list): List of column names (variables) to impute missing values for.

    Methods:
        fit(X, y=None):
            This method does not perform any actual training or fitting.
            It returns the transformer instance itself.

        transform(X):
            Imputes missing values in the specified categorical variables and returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    imputer = CategoricalImputer(variables=['category1', 'category2'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('imputer', imputer),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """
    def __init__(self, variables=None):
        """
        Initialize the CategoricalImputer transformer.

        Parameters:
            variables (list or str, optional): List of column names (variables) to impute missing values for.
                If a single string is provided, it will be treated as a single variable. Default is None.
        """
        self.variables = [variables] if not isinstance(variables, list) else variables

    def fit(self, X, y=None):
        """
        This method does not perform any actual training or fitting, as imputation is based on data.
        It returns the transformer instance itself.

        Parameters:
            X (pd.DataFrame): Input data to be transformed. Not used in this method.
            y (pd.Series or np.array, optional): Target variable. Not used in this method.

        Returns:
            self (CategoricalImputer): The transformer instance.
        """
        return self

    def transform(self, X):
        """
        Imputes missing values in the specified categorical variables and returns the modified DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame with missing values imputed for the specified categorical variables.
        """
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna('Missing')
        return X
    
class NumericalImputer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to impute missing values in numerical variables.

    Parameters:
        variables (list or str, optional): List of column names (variables) to impute missing values for.
            If a single string is provided, it will be treated as a single variable. Default is None.

    Attributes:
        variables (list): List of column names (variables) to impute missing values for.
        median_dict_ (dict): Dictionary to store the median values for each specified numerical variable during fitting.

    Methods:
        fit(X, y=None):
            Calculates the median values for the specified numerical variables from the training data.
            It returns the transformer instance itself.

        transform(X):
            Imputes missing values in the specified numerical variables using the median values and returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    imputer = NumericalImputer(variables=['age', 'income'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('imputer', imputer),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """
    def __init__(self, variables=None):
        """
        Initialize the NumericalImputer transformer.

        Parameters:
            variables (list or str, optional): List of column names (variables) to impute missing values for.
                If a single string is provided, it will be treated as a single variable. Default is None.
        """
        self.variables = [variables] if not isinstance(variables, list) else variables

    def fit(self, X, y=None):
        """
        Calculates the median values for the specified numerical variables from the training data.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            self (NumericalImputer): The transformer instance.
        """
        self.median_dict = {}
        for var in self.variables:
            self.median_dict[var] = X[var].median()
        return self


    def transform(self, X):
        """
        Imputes missing values in the specified numerical variables using the median values and returns the modified DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame with missing values imputed for the specified numerical variables.
        """
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna(self.median_dict[var])
        return X
    
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to encode rare categories in categorical variables.

    Parameters:
        tol (float, optional): The tolerance level to define rare categories.
            Categories with a frequency lower than tol will be encoded as 'rare'.
            Default is 0.05.
        variables (list or str, optional): List of column names (variables) to encode rare categories for.
            If a single string is provided, it will be treated as a single variable. Default is None.

    Attributes:
        tol (float): The tolerance level to define rare categories.
        variables (list): List of column names (variables) to encode rare categories for.
        rare_labels_dict (dict): Dictionary to store the rare category labels for each specified categorical variable during fitting.

    Methods:
        fit(X, y=None):
            Calculates the rare category labels for the specified categorical variables from the training data.
            It returns the transformer instance itself.

        transform(X):
            Encodes rare categories in the specified categorical variables and returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    encoder = RareLabelCategoricalEncoder(tol=0.1, variables=['category1', 'category2'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('encoder', encoder),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """
    def __init__(self, tol=0.05, variables=None):
        """
        Initialize the RareLabelCategoricalEncoder transformer.

        Parameters:
            tol (float, optional): The tolerance level to define rare categories.
                Categories with a frequency lower than tol will be encoded as 'rare'.
                Default is 0.05.
            variables (list or str, optional): List of column names (variables) to encode rare categories for.
                If a single string is provided, it will be treated as a single variable. Default is None.
        """
        self.tol = tol
        self.variables = [variables] if not isinstance(variables, list) else variables

    def fit(self, X, y=None):
        """
        Calculates the rare category labels for the specified categorical variables from the training data.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            self (RareLabelCategoricalEncoder): The transformer instance.
        """
        self.rare_labels_dict = {}
        for var in self.variables:
            t = pd.Series(X[var].value_counts() / float(X.shape[0]))
            self.rare_labels_dict[var] = list(t[t<self.tol].index)
        return self

    def transform(self, X):
        """
        Encodes rare categories in the specified categorical variables and returns the modified DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame with rare categories encoded for the specified categorical variables.
        """
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.rare_labels_dict[var]), 'rare', X[var])
        return X
      
class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to perform one-hot encoding for categorical variables.

    Parameters:
        variables (list or str, optional): List of column names (variables) to perform one-hot encoding for.
            If a single string is provided, it will be treated as a single variable. Default is None.

    Attributes:
        variables (list): List of column names (variables) to perform one-hot encoding for.
        dummies (list): List of column names representing the one-hot encoded dummy variables.

    Methods:
        fit(X, y=None):
            Calculates the one-hot encoded dummy variable columns for the specified categorical variables from the training data.
            It returns the transformer instance itself.

        transform(X):
            Performs one-hot encoding for the specified categorical variables and returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    encoder = OneHotEncoder(variables=['category1', 'category2'])

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('encoder', encoder),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """
    def __init__(self, variables=None):
        """
        Initialize the OneHotEncoder transformer.

        Parameters:
            variables (list or str, optional): List of column names (variables) to perform one-hot encoding for.
                If a single string is provided, it will be treated as a single variable. Default is None.
        """
        self.variables = [variables] if not isinstance(variables, list) else variables

    def fit(self, X, y=None):
        """
        Calculates the one-hot encoded dummy variable columns for the specified categorical variables from the training data.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            self (OneHotEncoder): The transformer instance.
        """
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        return self

    def transform(self, X):
        """
        Performs one-hot encoding for the specified categorical variables and returns the modified DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame with one-hot encoded dummy variables for the specified categorical variables.
        """
        X = X.copy()
        X = pd.concat([X, pd.get_dummies(X[self.variables], drop_first=True)], axis=1)
        X.drop(self.variables, axis=1)

        # Adding missing dummies, if any
        missing_dummies = [var for var in self.dummies if var not in X.columns]
        if len(missing_dummies) != 0:
            for col in missing_dummies:
                X[col] = 0

        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to select specific features (columns) from a DataFrame.

    Parameters:
        feature_names (list or array-like): List of column names to select as features from the input DataFrame.

    Methods:
        fit(X, y=None):
            Placeholder method that returns the transformer instance itself.

        transform(X):
            Selects and returns the specified features (columns) from the input DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Define the feature names to be selected
    selected_features = ['feature1', 'feature2', 'feature3']

    # Instantiate the custom transformer
    feature_selector = FeatureSelector(feature_names=selected_features)

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('feature_selector', feature_selector),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """

    def __init__(self, feature_names):
        """
        Initialize the FeatureSelector transformer.

        Parameters:
            feature_names (list or array-like): List of column names to select as features from the input DataFrame.
        """
        self.feature_names = feature_names

    def fit(self, X, y=None):
        """
        Placeholder method that returns the transformer instance itself.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            self (FeatureSelector): The transformer instance.
        """
        return self

    def transform(self, X):
        """
        Selects and returns the specified features (columns) from the input DataFrame.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_selected (pd.DataFrame): DataFrame containing only the specified features (columns).
        """
        return X[self.feature_names]

class OrderingFeatures(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to order features (columns) in the same order as they appeared in the training data.

    Parameters:
        None

    Attributes:
        ordered_features (pd.Index): Index of column names representing the order of features as they appeared in the training data.

    Methods:
        fit(X, y=None):
            Records the order of features from the training data and returns the transformer instance itself.

        transform(X):
            Reorders the features in the same order as they appeared in the training data and returns the modified DataFrame.

    Example usage:
    ```
    from sklearn.pipeline import Pipeline

    # Instantiate the custom transformer
    feature_orderer = OrderingFeatures()

    # Define the pipeline with the custom transformer
    pipeline = Pipeline([
        ('feature_orderer', feature_orderer),
        # Other pipeline steps...
    ])

    # Fit and transform the data using the pipeline
    X_transformed = pipeline.fit_transform(X)
    ```
    """
    def __init__(self):
        """
        Initialize the OrderingFeatures transformer.

        Parameters:
            None
        """
        return None

    def fit(self, X, y=None):
        """
        Records the order of features from the training data.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            self (OrderingFeatures): The transformer instance.
        """
        if isinstance(X, pd.DataFrame):
            self.ordered_features = X.columns
            print(self.ordered_features)
        elif isinstance(X, np.ndarray):
            self.ordered_features = np.arange(X.shape[1])
        else:
            raise ValueError("Input X must be a pandas DataFrame or a numpy array.")
        return self

    def transform(self, X):
        """
        Reorders the features in the same order as they appeared in the training data.

        Parameters:
            X (pd.DataFrame): Input data to be transformed.

        Returns:
            X_transformed (pd.DataFrame): Transformed DataFrame with features ordered as they appeared in the training data.
        """

        if isinstance(X, pd.DataFrame):
            # print(X[self.ordered_features])
            # print("return df")
            DROP_COLS_AFTER = ['no_of_adults','no_of_children','no_of_weekend_nights','no_of_week_nights','type_of_meal_plan','required_car_parking_space','room_type_reserved','lead_time','arrival_year','arrival_month','arrival_date','market_segment_type','repeated_guest','no_of_previous_cancellations','no_of_previous_bookings_not_canceled','avg_price_per_room','no_of_special_requests']
            X[self.ordered_features]
            X.drop(DROP_COLS_AFTER, axis=1, inplace=True)
            return X
        elif isinstance(X, np.ndarray):
            # print("return np")
            return X[:, self.ordered_features]
        else:
            raise ValueError("Input X must be a pandas DataFrame or a numpy array.")
