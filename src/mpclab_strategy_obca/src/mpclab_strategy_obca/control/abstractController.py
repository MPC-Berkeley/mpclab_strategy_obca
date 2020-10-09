#!/usr/bin python3

from abc import ABC, abstractmethod

class abstractController(ABC):

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def solve(self):
        pass
