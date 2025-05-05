from sklearn.base import BaseEstimator, TransformerMixin

class PreprocessDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if "patient_id" in X.columns:
            X = X.drop(columns=["patient_id"])
        return X     