# Python-predict-agent

## TODO
  v 1d feature, discrete action FeatureExtractorInverseNetwork
  v average loss
  v Save
  v Load
  V GPU
  V pygame
  - Load each
  
  - Jan 3rd
  - PredictorNetwork
  - ControllerNetwork
  - CircularEnvironment

  - Jan 4th
  - ComplexEnvironment
  - 2d FeatureExtractorInverseNetwork
  - Continuous action

  - Others
  - Fine tuning
  - Name to predictive-agent
  - Makefile setting
    - ref. https://kimjingo.tistory.com/203
  - LSTM
  - parallel modules
  - Best checkpoint
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