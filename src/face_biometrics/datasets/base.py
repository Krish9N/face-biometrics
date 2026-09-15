from abc import ABC, abstractmethod
from pathlib import Path

from .types import FaceSample


class FaceDataset(ABC):

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> FaceSample:
        pass