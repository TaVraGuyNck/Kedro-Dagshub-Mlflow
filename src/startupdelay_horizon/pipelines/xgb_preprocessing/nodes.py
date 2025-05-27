import mlflow
import pandas as pd


class XGBoostPreprocessor:
    def __init__(self, top_n_countries=10, top_n_funding=10):
        self.top_n_countries = top_n_countries
        self.top_n_funding = top_n_funding
        self.top_countries_ = None
        self.top_funding_ = None

    def fit(self, X, y=None):
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

    def transform(self, X):
        X = X.copy()

        #Fill missing values
        X["pillar"] = X["pillar"].fillna("missing")
        X["countryCoor"] = X["countryCoor"].fillna("missing")
        X["fundingScheme"] = X["fundingScheme"].fillna("missing")

        #Encode pillar as category codes
        X["pillar_encoded"] = X["pillar"].astype("category").cat.codes

        #Handle country
        X["country_clean"] = X["countryCoor"].where(
            X["countryCoor"].isin(self.top_countries_), "Other"
        )
        country_dummies = pd.get_dummies(X["country_clean"], prefix="country")

        #Handle funding scheme
        X["funding_clean"] = X["fundingScheme"].where(
            X["fundingScheme"].isin(self.top_funding_), "Other"
        )
        funding_dummies = pd.get_dummies(X["funding_clean"], prefix="funding")

        # Combine
        X = pd.concat([
            X.drop(columns=[
                "pillar", "countryCoor", "country_clean",
                "fundingScheme", "funding_clean"
            ]),
            country_dummies,
            funding_dummies
        ], axis=1)

        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def apply_xgb_transformer(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Applies XGBoost-specific preprocessing to already-split data and logs metadata to MLflow.
    Returns transformed train/test sets and the fitted preprocessor.
    """
    # No mlflow.start_run() here—run is managed by hooks.py!
    mlflow.set_tag("pipeline_step", "xgb_preprocessing_node")

    transformer = XGBoostPreprocessor()
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    mlflow.log_param("top_countries", transformer.top_countries_)
    mlflow.log_param("top_funding", transformer.top_funding_)

    return X_train_transformed, X_test_transformed, transformer
