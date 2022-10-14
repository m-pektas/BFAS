

import importlib
from ..archs import ArchFactory



class TestArchFactory:

    archname = "arch_1"
    archs_dir = "archs"
    arch_factory = ArchFactory(archs_dir)

    def test_initNetwork(cls):
        #check importing
        try:
            module = importlib.import_module(f"{cls.archs_dir}.{cls.archname}")
        except ImportError:
            raise ImportError("Arch module cannot be imported !!")

    def test_createMetaArch(cls):
        try:
            TestArchFactory.arch_factory.createMetaArch(cls.archname)
        except Exception as e:
            raise e


    def test_readSearchSpaceInfo(cls):
        try:
            TestArchFactory.arch_factory.readSearchSpaceInfo(cls.archname)
        except Exception as e:
            raise e

    
