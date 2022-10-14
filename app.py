
from bfas import BFAS
from bfas.utils.data_types import *

bfas = BFAS(project_name="bfas",
            archname="mobilenetv2_custom",
            archs_dir = "archs",
            device="cpu",
            logger_name="tensorboard",
            is_logging_active=True,
            seed=28,
            )

bfas.rules.addRule(Rule(Metrics.FPS, Limit.MIN, 1))
bfas.rules.addRule(Rule(Metrics.PARAMCOUNT, Limit.MAX, 10))
bfas.run(iter_count=5)
