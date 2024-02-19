# Python-predict-agent

## TODO
- CircularEnvironment
- Continuous action
- Implement VBN-like, curiosity-like, world-model-like
- Put random policy an entropy

- Makefile setting (ref. https://kimjingo.tistory.com/203)
- Parallel modules
- Rethinking loss at once
- Theoretical proof of convergence

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