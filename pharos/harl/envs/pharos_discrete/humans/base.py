from abc import ABC, abstractmethod
import numpy as np

class Human:
    def __init__(self, id: str, scare_factor: float,
                 position: np.ndarray = None, velocity: np.ndarray = None):
        """
        :param id:
        :param name:
        :param scare_factor: 恐惧因子
        """
        self.id = id
        self.scare_factor = scare_factor
        self.position = position if position is not None else np.zeros(3, dtype=float)
        self.velocity = velocity if velocity is not None else np.zeros(3, dtype=float)
    
    @abstractmethod
    def update(self, delta_t: float) -> None:
        """
        更新人类的状态
        """
        pass
    
    def __repr__(self):
        return f"Human(id={self.id}, scare_factor={self.scare_factor})"
    def __str__(self):
        return f"Human {self.id} (scare_factor={self.scare_factor})"
    def __eq__(self, other):
        if isinstance(other, Human):
            return self.id == other.id
        return False
    def __hash__(self):
        return hash(self.id) 
        