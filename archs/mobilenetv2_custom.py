import torchvision.models as models
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.mobilenet = models.mobilenet_v2(weights=None)
        
        
        classifier_option_1 = nn.Sequential(nn.Linear(1280, args["linear1_out"]),
                                                   nn.Dropout(p=args["dropout1_p"]),
                                                   nn.Linear(args["linear1_out"], args["linear2_out"]),
                                                   nn.Dropout(p=args["dropout1_p"]),
                                                   nn.Linear(args["linear2_out"], args["num_classes"]),
                                                   )

        classifier_option_2 = nn.Sequential(nn.Linear(1280, args["linear1_out"]),
                                             nn.Dropout(p=args["dropout1_p"]),
                                             nn.Linear(args["linear1_out"], args["num_classes"]))
        
        if args["classifier_option"] == 1:
            self.mobilenet.classifier = classifier_option_1
        elif args["classifier_option"] == 2:
            self.mobilenet.classifier = classifier_option_2
        else:
            raise NotImplementedError("NotImplementedError")
            

    def forward(self, x):
        x = self.mobilenet(x)
        return x

    def input_producer(self, bs=1):
        device = next(self.parameters()).device
        x = torch.FloatTensor(bs, 3, 224, 224).to(device)
        return {"x":x}