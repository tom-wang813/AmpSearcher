from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from amp.model.ml.base import ModelModule

class SVM(ModelModule):

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        probability: bool = True,
        class_weight=None,
        random_state=None,
        **kwargs,
    ):

        """
        超参数全部由外部传入，取消内部对 cfg 的依赖。
        可直接 SVM(C=0.5, kernel='linear') 调用。
        """
        super().__init__(
            C=C,
            kernel=kernel,
            probability=probability,
            class_weight=class_weight,
            random_state=random_state,
            **kwargs,
        )

    def build(self):
        # 从 base.params 取出非 None 的超参
        filtered = {k: v for k, v in self.params.items() if v is not None}
        self.model = SVC(**filtered)
        return self.model


    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        res = {"accuracy": accuracy_score(y, y_pred)}
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            res["roc_auc"] = roc_auc_score(y, scores)
        elif hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
            res["roc_auc"] = roc_auc_score(y, probs)
        return res