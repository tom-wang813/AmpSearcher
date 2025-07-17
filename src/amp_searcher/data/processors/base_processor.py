from abc import ABC, abstractmethod
from typing import List, Any


class BaseProcessor(ABC):
    @abstractmethod
    def process(self, data: List[str]) -> List[Any]:
        """
        Abstract method to process a list of data items.

        Args:
            data: A list of data items (e.g., peptide sequences).

        Returns:
            A list of processed data items (e.g., featurized sequences).
        """
        pass

    @abstractmethod
    def process_single(self, data_item: str) -> Any:
        """
        Abstract method to process a single data item.

        Args:
            data_item: A single data item (e.g., a peptide sequence).

        Returns:
            A single processed data item (e.g., a featurized sequence).
        """
        pass
