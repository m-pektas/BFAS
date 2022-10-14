from enum import Enum, auto
from dataclasses import dataclass
from torch import nn
import numpy as np

class Limit(Enum):
    MIN = auto()
    MAX = auto()

class Metrics(Enum):
    GMAC =  "gmac"
    PARAMCOUNT = "param_count"
    FPS = "fps"

class SearchStrategy(Enum):
    GRID =  auto()
    RANDOM = auto()

@dataclass
class Rule:
    metric: Metrics
    limit : Limit
    val : float

class Rules():
    def __init__(self):
        self.rules = []

    def addRule(self, rule: Rule):
        self.rules.append(rule)

    def checkRules(self, rules_metrics):
        result = True
        for rule in self.rules:
            if rule.limit == Limit.MAX and rules_metrics[rule.metric.value] > rule.val:
                result = False
                break
            if rule.limit == Limit.MIN and rules_metrics[rule.metric.value] < rule.val:
                result = False
                break

        return result
    


class MetaArch:
    def __init__(self, network : nn.Module, search_space_info : dict ):
        self.network = network
        self.__search_space_info = search_space_info
        self.search_space = self.initSearchSpace()
    
    def initSearchSpace(self) -> dict:
        """
        Create parameter space using parameter informations that are readed from architecture json file.
        """
        space = {}
        for key, val in self.__search_space_info.items():
            if val["type"] == "range":
                space[key] = np.arange(
                    val["range"][0], val["range"][1], val["step"])
            elif val["type"] == "array":
                space[key] = val["values"]
        return space

    def getParams(self) -> dict:
        """
        This function sample the params of the network from parameter space.
        """
        param = {}
        for k, v in self.search_space.items():
            param[k] = np.random.choice(self.search_space[k])
        return param

    def getModel(self, param, device):
        return self.network(param).eval().to(device)

  
