import pandas as pd
from catboost import CatBoostRegressor
import mlflow


class CatBoostPreprocessor:
    def __init__(self, top_n_countries=10, top_n_funding=10):
        self.top_n_countries = top_n_countries
        self.top_n_funding = top_n_funding
        self.top_countries_ = None
        self.top_funding_ = None
        self.cat_features = []

    def fit(self, X: pd.DataFrame, y=None):
        self.top_countries_ = (
            X["countryCoor"]
            .fillna("missing")
            .value_counts()
            .nlargest(self.top_n_countries)
            .index
            .tolist()
        )
        self.top_funding_ = (
            X["fundingScheme"]
            .fillna("missing")
            .value_counts()
            .nlargest(self.top_n_funding)
            .index
            .tolist()
        )
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        # Fill missing values
        X["pillar"] = X["pillar"].fillna("missing")
        X["countryCoor"] = X["countryCoor"].fillna("missing")
        X["fundingScheme"] = X["fundingScheme"].fillna("missing")

        # Reduce cardinality
        X["countryCoor"] = X["countryCoor"].where(
            X["countryCoor"].isin(self.top_countries_), "Other"
        )
        X["fundingScheme"] = X["fundingScheme"].where(
            X["fundingScheme"].isin(self.top_funding_), "Other"
        )

        # Convert to strings
        X["pillar"] = X["pillar"].astype(str)
        X["countryCoor"] = X["countryCoor"].astype(str)
        X["fundingScheme"] = X["fundingScheme"].astype(str)

        return X

    def fit_transform(self, X: pd.DataFrame, y=None):
        self.fit(X, y)
        transformed = self.transform(X)
        self.cat_features = ["pillar", "countryCoor", "fundingScheme"]
        return transformed


# --- STEP 3: Apply transformer ---
def apply_cb_transformer(X_train: pd.DataFrame, X_test: pd.DataFrame):
        mlflow.set_tag("pipeline_step", "cb_preprocessing_node")

        transformer = CatBoostPreprocessor()
        X_train_transformed = transformer.fit_transform(X_train)
        X_test_transformed = transformer.transform(X_test)

        transformer.feature_names_ = X_train_transformed.columns.tolist()

        mlflow.set_tag("cat_features", ", ".join(transformer.cat_features))
        mlflow.log_param("top_countries", transformer.top_countries_)
        mlflow.log_param("top_funding", transformer.top_funding_)

        return (
            X_train_transformed,
            X_test_transformed,
            transformer.cat_features,
            transformer
        )

