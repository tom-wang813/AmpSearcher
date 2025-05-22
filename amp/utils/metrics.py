from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        """清空内部状态，例如计数器、累加值等"""
        pass

    @abstractmethod
    def update(self, y_true, y_pred):
        """接收一批真实值和预测值，用来累积计算"""
        pass

    @abstractmethod
    def compute(self):
        """返回当前累积状态下的最终指标值"""
        pass


class Accuracy(Metric):
    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, y_true, y_pred):
        # 假设 y_true, y_pred 是 numpy 数组或可以用 == 比较
        self.correct += (y_true == y_pred).sum()
        self.total += y_true.shape[0]

    def compute(self):
        return self.correct / self.total if self.total else 0.0


class MSE(Metric):
    def reset(self):
        self.sum_sq_err = 0.0
        self.n = 0

    def update(self, y_true, y_pred):
        import numpy as np
        err = (y_true - y_pred) ** 2
        self.sum_sq_err += err.sum()
        self.n += y_true.shape[0]

    def compute(self):
        return self.sum_sq_err / self.n if self.n else 0.0


class MultiTaskMetric(Metric):
    def __init__(self, metric_dict):
        """
        metric_dict: {'task1': Metric(), 'task2': Metric(), ...}
        """
        self.metric_dict = metric_dict
        super().__init__()

    def reset(self):
        for m in self.metric_dict.values():
            m.reset()

    def update(self, y_true_dict, y_pred_dict):
        # y_true_dict, y_pred_dict: {'task1': ..., 'task2': ...}
        for task, m in self.metric_dict.items():
            m.update(y_true_dict[task], y_pred_dict[task])

    def compute(self):
        return {task: m.compute() for task, m in self.metric_dict.items()}