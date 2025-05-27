from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from .base_dataloader import BaseDataLoader, BaseDataset, IterableBaseDataset

class MultiTaskDataset(BaseDataset):
    """多任务数据集，支持多输入多输出"""
    def __init__(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        input_transforms: Optional[Dict[str, Callable]] = None,
        target_transforms: Optional[Dict[str, Callable]] = None
    ):
        """
        Args:
            inputs: 输入数据字典，格式为 {'input_name': tensor}
            targets: 目标数据字典，格式为 {'target_name': tensor}
            input_transforms: 输入数据转换函数字典
            target_transforms: 目标数据转换函数字典
        """
        super().__init__()
        # 验证所有输入张量的第一维（样本数）相同
        lengths = [tensor.size(0) for tensor in inputs.values()]
        assert len(set(lengths)) == 1, "All inputs must have the same number of samples"
        
        self.inputs = inputs
        self.targets = targets
        self.input_transforms = input_transforms or {}
        self.target_transforms = target_transforms or {}
        self._length = lengths[0]
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """返回指定索引的数据样本"""
        batch = {}
        
        # 处理输入数据
        for name, tensor in self.inputs.items():
            value = tensor[index]
            if name in self.input_transforms:
                value = self.input_transforms[name](value)
            batch[f"input_{name}"] = value
        
        # 处理目标数据
        for name, tensor in self.targets.items():
            value = tensor[index]
            if name in self.target_transforms:
                value = self.target_transforms[name](value)
            batch[f"target_{name}"] = value
            
        return batch

class MultiTaskIterableDataset(IterableBaseDataset):
    """可迭代的多任务数据集，支持动态加载"""
    def __init__(
        self,
        data_generator,
        input_names: List[str],
        target_names: List[str],
        input_transforms: Optional[Dict[str, Callable]] = None,
        target_transforms: Optional[Dict[str, Callable]] = None
    ):
        """
        Args:
            data_generator: 数据生成器函数，每次调用返回 (inputs, targets)
            input_names: 输入数据的名称列表
            target_names: 目标数据的名称列表
            input_transforms: 输入数据转换函数字典
            target_transforms: 目标数据转换函数字典
        """
        super().__init__()
        self.data_generator = data_generator
        self.input_names = input_names
        self.target_names = target_names
        self.input_transforms = input_transforms or {}
        self.target_transforms = target_transforms or {}
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # 如果是多线程工作者，使用worker_id来区分数据
            worker_id = worker_info.id
        else:
            # 如果是主线程，worker_id为0
            worker_id = 0

        print(f"Worker {worker_id} started iterating over dataset")
        while True:
            # 获取下一批数据
            inputs, targets = self.data_generator(worker_id)
            batch = {}
            
            # 处理输入数据
            for name, value in zip(self.input_names, inputs):
                if name in self.input_transforms:
                    value = self.input_transforms[name](value)
                batch[f"input_{name}"] = value
            
            # 处理目标数据
            for name, value in zip(self.target_names, targets):
                if name in self.target_transforms:
                    value = self.target_transforms[name](value)
                batch[f"target_{name}"] = value
                
            yield batch

class MultiTaskDataLoader(BaseDataLoader):
    """多任务数据加载器"""
    def __init__(
        self,
        dataset: Union[MultiTaskDataset, MultiTaskIterableDataset],
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn
        )
