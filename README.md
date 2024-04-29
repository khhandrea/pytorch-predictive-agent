# Python-predict-agent

## TODO
- Shorter term
  - Train code abstraction (from DML to real)
  - Synchronize with DML
- Longer term
  - CircularEnvironment
  - Continuous action
  - Implement VBN-like, curiosity-like, world-model-like
  - Makefile setting (ref. https://kimjingo.tistory.com/203)
  - Theoretical proof of convergence

## Usage
- Train
```
python3 main.py --config configs/test.yaml
```

- Tensorboard
```
tensorboard --logdir=experiment_results
(or)
tensorboard --logdir=experiment_results/{experiment name}
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
  name = "predictive-agent-test",
  srcs = ["python-predictive-agent/dml_trainer_test.py"],
  data = [":deepmind_lab.so"],
  main = "python-predictive-agent/dml_trainer_test.py"
)

py_binary(
  name = "predictive-agent",
  srcs = ["python-predictive-agent/dml_trainer.py"],
  data = [":deepmind_lab.so"],
  main = "python-predictive-agent/dml_trainer.py"
)
```

Run the bazel command to run the agent
```shell
bazel run :predictive-agent-test
(or)
bazel run :predictive-agent
```

(reference: https://github.com/miyosuda/unreal?tab=readme-ov-file)