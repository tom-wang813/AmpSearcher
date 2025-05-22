from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from amp.model.ml.base import ModelModule

class RF(ModelModule):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "gini",
        random_state: int = 42,
        **kwargs,
    ):
        """
        超参数全部由外部传入，不直接依赖 cfg。
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            criterion=criterion,
            random_state=random_state,
            **kwargs,
        )
        self.model = self.build()

    def build(self):
        # 从 base.params 取出非 None 的超参
        filtered = {k: v for k, v in self.params.items() if v is not None}
        self.model = RandomForestClassifier(**filtered)
        return self.model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        res = {"accuracy": accuracy_score(y, y_pred)}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
            res["roc_auc"] = roc_auc_score(y, probs)
        return res