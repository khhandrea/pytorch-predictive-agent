# Python-predict-agent

## TODO
  - 1d feature, discrete action FeatureExtractorInverseNetwork
    - forward function
    - Save and load
    - Fine tuning
  - makefile setting
    - ref. https://kimjingo.tistory.com/203
  - Rethinking loss at once
  - PredictorNetwork
  - 1d FeatureExtractorInverseNetwork
  - ControllerNetwork
  - CircularEnvironment
  - ComplexEnvironment
  - 2d FeatureExtractorInverseNetwork
  - continuous action

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