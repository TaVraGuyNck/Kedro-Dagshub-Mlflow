import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def fit_transform_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Applies scaling to numeric columns and one-hot encoding to categoricals.
    """
    columns_to_scale = [
        "totalCost", 
        "ecMaxContribution", 
        "duration", 
        "contRatio", 
        "numberOrg"
    ]
    columns_to_encode = [
        "pillar", 
        "countryCoor", 
        "fundingScheme"
    ]
    
    transformer = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), columns_to_scale),
            ("dummies", OneHotEncoder(handle_unknown="ignore", sparse_output=False), columns_to_encode),
        ],
        remainder="drop"
    )
    
    X_train_trans = transformer.fit_transform(X_train)
    X_test_trans = transformer.transform(X_test)

    # Get feature names from the transformer
    scale_features = columns_to_scale
    # Get names for one-hot encoded features
    ohe = transformer.named_transformers_["dummies"]
    ohe_features = ohe.get_feature_names_out(columns_to_encode)
    feature_names = list(scale_features) + list(ohe_features)

    X_train_trans = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_test_trans = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    return X_train_trans, X_test_trans, transformer
