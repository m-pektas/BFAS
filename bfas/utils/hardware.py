from cpuinfo import get_cpu_info
import torch

def getDeviceInfo(device):
    info = None
    if device == "cpu":
        try:
            info = get_cpu_info()["brand_raw"]
        except:
            try:
                info = get_cpu_info()["arch"]
            except:
                print("[W] Cannot reading device informations !!")

      
    elif torch.cuda.is_available():
        try:
            info = torch.cuda.get_device_name(0)
        except:
            print("Cannot reading device informations !!")
    elif device == "mps":
        info = "apple M series"
    else:
        raise ValueError("Unsupported device type !!")

    return info