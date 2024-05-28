# Python-predict-agent

## TODO
- Shorter term
  - Train code abstraction (from DML to real)
- Longer term
  - CircularEnvironment
  - Continuous action
  - Implement VBN-like, curiosity-like, world-model-like
  - Makefile setting (ref. https://kimjingo.tistory.com/203)
  - Theoretical proof of convergence

## main.py
- Train
```
python3 main.py --config configs/test.yaml
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
package(default_visibility = ["//visibility:public"])

py_library(
    name = "prednav-lib",
    srcs = glob(["python_predictive_agent/**/*.py"]),
    data = [":deepmind_lab.so"],
    imports = ["python_predictive_agent"]
)

filegroup(
    name = "prednav-filegroup",
    srcs = glob(["python_predictive_agent/configs/*.yaml"])
)

py_binary(
    name = "prednav",
    srcs = ["python_predictive_agent/main_dml.py"],
    data = [
        ":deepmind_lab.so",
        ":prednav-filegroup"
    ],
    main = "python_predictive_agent/main_dml.py",
    deps = [
        ":prednav-lib",
    ]
)
```

Run the bazel command to run the agent
```shell
bazel run :prednav
```
or from user windows
```shell
bazel run :prednav --define graphics=sdl
```

## Results
| name | description |
| --- | --- |
| config.yaml | configuration file of the experiment |
| checkpoints/ | model parameters, the form of state_dict |
| coordinates/ | agent trajectories, the form of csv |
| log/ | tensorboard log |

## Tensorboard
```
tensorboard --logdir=experiment_results(/{experiment name})
```