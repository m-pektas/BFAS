from time import time
import numpy as np
from thop import profile


def getComplexityMetrics(model):
    macs, params = profile(model, inputs=(model.input_producer()["x"], ), verbose=False)
    gmacs, params = macs/100000000, params/1000000
    return gmacs, params

# def getParamCount(model):
   
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad: continue
#         params = parameter.numel()
#         total_params+=params

#     total_params_M = total_params/1000000
#     return total_params_M

def getFPS(model, input, iter):
    times = []
    for i in range(iter):
        s = time()
        model(**input)
        duration = time()-s
        times.append(duration)

    return 1/np.mean(times)


