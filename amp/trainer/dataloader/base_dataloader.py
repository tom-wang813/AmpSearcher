from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

class BaseDataset(Dataset):
    """基础数据集类"""
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __len__(self) -> int:
        pass
        
    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """返回单个数据样本，格式为 {'input_name': tensor, 'target_name': tensor}"""
        pass

class IterableBaseDataset(IterableDataset):
    """基础可迭代数据集类，用于动态加载数据"""
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """返回数据迭代器，每个元素格式为 {'input_name': tensor, 'target_name': tensor}"""
        pass

class BaseDataLoader:
    """基础数据加载器类"""
    def __init__(
        self,
        dataset: Union[BaseDataset, IterableBaseDataset],
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 如果是IterableDataset，则不能shuffle
        if isinstance(dataset, IterableDataset):
            shuffle = False
            
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn or self.default_collate_fn
        )
    
    def default_collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """默认的数据批处理函数"""
        result = {}
        for key in batch[0].keys():
            result[key] = torch.stack([b[key] for b in batch])
        return result
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
