#!/usr/bin python3

from abc import ABC, abstractmethod

class abstractStrategyPredictor(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def predict(self):
        pass
