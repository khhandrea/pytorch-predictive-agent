# Python-predict-agent

## TODO
- Jan 2nd
  - v 1d feature, discrete action FeatureExtractorInverseNetwork
  - v average loss
  - v Save
  - v Load
  - v GPU
  - v pygame
  - v Load each
  - v PredictorNetwork
  - v Overfitting

- Jan 3rd
  - v ControllerNetwork
  - CircularEnvironment

- Jan 4th
  - v Read deepmind-lab
  - v ComplexEnvironment
  - v 2d FeatureExtractorInverseNetwork
  - Continuous action

- Feb 1st
  - Make inverse network module
  - Network configuration
  - Save and load

  - Implement VBN-like, curiosity-like, 

- Others
  - v Fine tuning
  - v Name to predictive-agent
  - Makefile setting
    - ref. https://kimjingo.tistory.com/203
  - LSTM to icm
  - parallel modules
  - Best checkpoint
  - Rethinking loss at once
  - Theoretical proof of convergence
  - prev_input -> prev_feature with retain_graph=True?

## Usage
- Train help
```
python3 main.py --help
```

- Train (example)
```
python3 main.py --random_policy
```

- Tensorboard
```
sh sh/tensorboard.sh
```

## Deepmind lab
Download and install deepmind lab
```shell
git clone https://github.com.deepmind/lab.git
```

Build it following the build instructions

Clone repo inside the lab directory
```shell
cd lab
git clone https://github.com/khhandrea/python-predictive-agent
```

Add following lines at the end of lab/BUILD file
```python
py_binary(
  name = "predictive-agent",
  srcs = ["python-predictive-agent/agent.py"],
  data = [":deepmind_lab.so"],
  main = "python-predictive-agent/agent.py"
)
```

Run the bazel command to run the agent
```shell
bazel run :predictive-agent
```