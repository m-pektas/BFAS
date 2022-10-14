
from ..archs import ArchFactory
from  .. import utils

class Test_Utils_Complexity:
    
    archname = "arch_1"
    archs_dir = "archs"
    arch_factory = ArchFactory(archs_dir)

    def test_getComplexityMetrics(cls):
        try:
            metaArch = cls.arch_factory.createMetaArch(cls.archname)
            param = metaArch.getParams()
            model = metaArch.getModel(param, "cpu")
        except:
            raise Exception("Could not create metaArch object correctly")

        try:
            gmac, params = utils.complexity.getComplexityMetrics(model)
        except Exception as e:
            raise e

    def test_getFPS(cls):
        try:
            metaArch = cls.arch_factory.createMetaArch(cls.archname)
            param = metaArch.getParams()
            model = metaArch.getModel(param, "cpu")
        except:
            raise Exception("Could not create metaArch object")



        try:
            input = model.input_producer()
            fps = utils.complexity.getFPS(model, input, 10)
        except:
            raise ValueError("Complexity metrics (fps) cannot computing !!!")


class Test_Utils_DataTypes:
    
    archname = "arch_1"
    rule_metrics =  {
                        utils.data_types.Metrics.FPS.value: 10,
                        utils.data_types.Metrics.PARAMCOUNT.value: 5,
                        utils.data_types.Metrics.GMAC.value: 10,
                    }
    archs_dir = "archs"
    arch_factory = ArchFactory(archs_dir)

    def test_Rules(cls):
        try:
            rules = utils.data_types.Rules()

            rules.addRule(utils.data_types.Rule(utils.data_types.Metrics.FPS, 
                                          utils.data_types.Limit.MIN, 
                                          1))

            rules.addRule(utils.data_types.Rule(utils.data_types.Metrics.PARAMCOUNT, 
                                          utils.data_types.Limit.MAX, 
                                          10))

            
            rules.checkRules(rules_metrics=cls.rule_metrics)
        except Exception as e:
            raise e


    def test_MetaArch(cls):
        try:
            metaArch = cls.arch_factory.createMetaArch(cls.archname)
            param = metaArch.getParams()
            model = metaArch.getModel(param, "cpu")
        except:
            raise Exception("Could not create metaArch object correctly")