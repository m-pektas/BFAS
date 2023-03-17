
from .archs import ArchFactory
from .utils.data_types import *
from . import utils
from tqdm import tqdm
from .training.trainer import Trainer

class BFAS:
    """
    Brute Force Architecture Search class
    """

    def __init__(self, project_name: str,
                       archname: str, 
                       device: str, 
                       archs_dir : str, 
                       seed: int = None, 
                       logger_name : str = "wandb",
                       is_logging_active=True,
                       log_dir : str = "log",
                       is_training_active : bool = False,
                       dataset : str = "mnist",
                       batch_size : int = 32,
                       test_step : int = 20,
                       train_step : int = 100,
                       
                       
                       ) -> None:
        """
        project_name: The name of the project.
        archname: The name of the architecture files. It have to same with archname of archs/archname.json and archs/archname.py.
        logger_name: The name of the logger. You can use "tensorboard" and "wandb" loggers.
        is_logging_active: Whether the logger is active.
        log_dir: The directory where the log files will saved.
        """
        
        

        self.rules = Rules()
        self.archname = archname
        self.device = device
        self.project_name  = project_name
        self.device_name = utils.hardware.getDeviceInfo(device)
        self.seed = seed
        self.setSeed(seed)
        self.archfactory = ArchFactory(archs_dir)
        self.metaArch = self.archfactory.createMetaArch(archname)
        self.project_info = self.getProjectInfo()
        self.log = utils.logging.Log(self.project_info, logger_name, is_logging_active, log_dir)
        self.is_training_active = is_training_active
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_step = test_step
        self.train_step = train_step



        

    def run(self, iter_count: int):
        """ 
        This function runs searchig process.
        """
        
        is_solution_found = False
        for step in tqdm(range(iter_count), desc="Searchig ..."):

            # sample param and get model
            param = self.metaArch.getParams()
            model = self.metaArch.getModel(param, self.device)

            
   
            # measure other metrics
            fps = utils.complexity.getFPS(model, model.input_producer(), 5)
            gmacs, params = utils.complexity.getComplexityMetrics(model)

            metrics = {
                Metrics.FPS.value: fps,
                Metrics.PARAMCOUNT.value: params,
                Metrics.GMAC.value: gmacs,
            }

            # train and test model
            if self.is_training_active:
                trainer = Trainer(model, self.dataset, self.batch_size, self.device)
                trainer.train(self.train_step)
                loss, acc = trainer.test(self.test_step)
            
                metrics[Metrics.LOSS.value] = loss
                metrics[Metrics.ACC.value] = acc
               

            # if my rules is satisfied, log the results to logger.
            if self.rules.checkRules(metrics):
                is_solution_found = True
                self.log.logger.log_scaler(param, step)
                self.log.logger.log_scaler_metrics(metrics, step)
            

            if step > iter_count/2 and is_solution_found == False:
                self.log.logger.alert(
                    text=f"Half of searhing progress is finished but cannot find any solution !!"
                )

    def addRule(self, rule):
        self.rules.addRule(rule)

    def setSeed(self, seed):
        if not (seed is None):
            import numpy as np
            np.random.seed(seed)

    def getProjectInfo(self) -> dict:
        return {"name":self.project_name,
                "archname":self.archname,
                "device_name": self.device_name,
                "device":self.device,
                "seed": self.seed}