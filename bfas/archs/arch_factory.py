

import importlib
from dataclasses import dataclass
from torch import nn
from ..utils.data_types import MetaArch
from ..utils import fileio
import os

class ArchFactory:
    """
    This class create metaarch object. 
    This object stored given architecture and their seach space.
    """

    def __init__(self, archs_dir=None):
        self.archs_dir = archs_dir

    def readSearchSpaceInfo(self, name):
        search_space = None
        path = f"{self.archs_dir}/{name}"
        if os.path.exists(path + ".json"):
            search_space = fileio.readJson(path + ".json")
        elif os.path.exists(path + ".yaml"):
            search_space = fileio.readJson(path + ".yaml")
        else:
            message = f"Search space file not found for {name}"
            raise Exception(message)

        return search_space


    def initNetwork(self, name: str):
        modeule_path = self.archs_dir.replace("/",".")
        module = importlib.import_module(f"{modeule_path}.{name}")
        return module

    def createMetaArch(self, name: str):
        """
        This function create metaarch object.
        """
        network = self.initNetwork(name).Net
        search_space_info = self.readSearchSpaceInfo(name)
        return MetaArch(network, search_space_info)