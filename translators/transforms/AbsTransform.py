from torch import nn
from abc import ABC, abstractmethod

''' Concept Note : Abstract Class (for personal study)
Definition >
- A class that cannot be directly instantiated. 
- Serves as a blueprint for other classes.

==== [ MEMO ] ====
ABC in Python stands for Abstract Base Class, which is a module that 
provides tools for defining abstract classes.
'''

class AbsTransform(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.transform = None

    def forward(self, x):
        return self.transform(x)