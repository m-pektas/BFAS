
import wandb
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
import os 
from . import fileio


class Log():
    def __init__(self, project_info : str, logger_name : str, is_active : bool, log_dir : str):
        self.project_info = project_info
        self.logger_name = logger_name
        self.is_active = is_active
        self.log_dir = log_dir
        # self.set_local_logging()
        self.logger = self.init_logger()
        

    def init_logger(self):
        if self.is_active:
            if self.logger_name == "wandb":
                return WandbLogger(self.project_info, self.log_dir, self.is_active)
            elif self.logger_name == "tensorboard":
                return TBLogger(self.project_info, self.log_dir, self.is_active)
            else:
                raise ValueError("Invalid logger type !!")
        else:
            return DefaultLogger()

    def set_local_logging(self):
        if self.is_active:
            date = datetime. now(). strftime("%Y_%m_%d-%I:%M:%S_%p")
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(self.log_dir+'/'+date+'.log', 'w', 'utf-8')
            root_logger.addHandler(handler)
        

class ILog(ABC):
    
    @abstractmethod
    def init(name):
        pass

    @abstractmethod
    def log_scaler(scalers : dict, step : int):
        pass
    
    def alert(self, text):
        # logging.warning(text)
        print("| Warnining |",text)

class DefaultLogger(ILog):
    def init(self, name):
        pass
    
    def log_scaler(self, scalers : dict, step: int):
        pass
    
    def log_scaler_metrics(self, scalers : dict, step: int):
        pass

class TBLogger(ILog):
    def __init__(self, project_info, log_dir:str, is_active : bool):
        self.is_active = is_active
        self.logdir = log_dir
        self.project_info = project_info
        date = datetime.now(). strftime("%Y_%m_%d-%I:%M:%S_%p")
        self.log_dir_edited = f"{self.logdir}/tb_{project_info['archname']}_{date}"
        os.makedirs(self.log_dir_edited, exist_ok=True)
        fileio.writeJson(project_info, self.log_dir_edited+"/info.json" )
        self.init(self.log_dir_edited)
        # logging.info("Tensorboard logging is activated !!")

    def init(self, log_dir):
        if self.is_active:
            self.logger = SummaryWriter(log_dir)

    def log_scaler(self, scalers : dict, step : int):
        if self.is_active:
            for k, v in scalers.items():
                self.logger.add_scalar(k, v, step)
    
    def log_scaler_metrics(self, scalers : dict, step : int):
        
        if self.is_active:
            for k, v in scalers.items():
                v = scalers[k]
                k = "Metrics/"+k
                self.logger.add_scalar(k, v, step)
    
class WandbLogger(ILog):
    def __init__(self, project_info, log_dir : str, is_active : bool):
        self.is_active = is_active
        self.logdir = log_dir
        os.makedirs(self.logdir, exist_ok=True)
        self.init(project_info["name"], self.logdir)
        # logging.info("Wandb logging is activated !!")
        wandb.run.name = f"{project_info['archname']}_{wandb.run.id}"
        fileio.writeJson(project_info, wandb.run.dir+"/info.json" )


    def init(self, name, log_dir):
        if self.is_active:
            wandb.init(project=name, dir=log_dir)

    def log_scaler(self, scalers : dict, step : int):
        if self.is_active:
            wandb_dict = {}
            for k, v in scalers.items():
                wandb_dict["Params/"+k] = scalers[k]
            wandb.log(wandb_dict, step=step)

    def log_scaler_metrics(self, scalers : dict, step : int):
        if self.is_active:
            wandb_dict = {}
            for k, v in scalers.items():
                wandb_dict["Metrics/"+k] = scalers[k]
            wandb.log(wandb_dict, step=step)
