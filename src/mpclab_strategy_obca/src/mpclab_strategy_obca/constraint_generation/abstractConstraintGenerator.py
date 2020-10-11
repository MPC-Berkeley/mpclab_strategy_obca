#!/usr/bin python3

from abc import ABC, abstractmethod

class abstractConstraintGenerator(object):
    @abstractmethod
    def generate_constraint(self):
        pass
