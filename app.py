
from bfas import BFAS
from bfas.utils.data_types import *

bfas = BFAS(project_name="bfas",
            archname="resnet50_custom",
            archs_dir = "archs",
            device="mps",
            logger_name="tensorboard",
            is_logging_active=True,
            seed=28,
            is_training_active=True,
            dataset="cifar10",
            batch_size=32,
            test_step=50,
            train_step=250,
            )

bfas.rules.addRule(Rule(Metrics.FPS, Limit.MIN, 1))
bfas.rules.addRule(Rule(Metrics.PARAMCOUNT, Limit.MAX, 10))
bfas.run(iter_count=3)
