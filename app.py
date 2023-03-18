
from bfas import BFAS
from bfas.utils.data_types import *

bfas = BFAS(project_name="bfas",
            archname="resnet50_custom",
            archs_dir = "archs",
            device="mps",
            logger_name="wandb",
            is_logging_active=True,
            seed=25,
            is_training_active=True,
            dataset="cifar10",
            batch_size=16,
            test_step=5,
            train_step=50,
            )

bfas.rules.addRule(Rule(Metrics.FPS, Limit.MIN, 20))
bfas.rules.addRule(Rule(Metrics.PARAMCOUNT, Limit.MIN, 20))
bfas.run(iter_count=10)
