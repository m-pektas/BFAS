# Create Environment

## Default

```
conda create -n bfas python=3.8 -y
pip install -r bfas/env/requirements.txt
pip install .
```
## Docker

```bash
cd BFAS
docker build . -f bfas/env/Dockerfile -t bfas:v0.0.1
docker run -it -v ${PWD}:/app bfas:v0.0.1
```

# How can I use BFAS ?
### 0. Create your architecture folder 

```bash
mkdir archs
```
### 1. Implement your architecture and put it into your arhitecture folder. Your architecture should takes all variables as initialization parameters.

```python
# archs/archname.py
class Net(nn.Module):
    def __init__(self, args: dict):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(110*110*6, args["linear1_out"])
        self.fc2 = nn.Linear(args["linear1_out"], 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def input_producer(self, bs=1):
        x = torch.FloatTensor(bs, 3, 224, 224).cpu()
        return {"x":x}
```


### 2. Set your parameters space specifications and put it into your arhitecture folder.

```json
# archs/archname.json
{
  "linear1_out": {
    "type" : "range",
    "range" : [1, 1000],
    "step" : 10
  }
}
```


###  3. Create own experiment

```python
#app.py

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
```

### 4. Run your script

```bash
python app.py
```






