# Python-predict-agent

## TODO
  v 1d feature, discrete action FeatureExtractorInverseNetwork
  v average loss
  v Save
  v Load
  V GPU
  V pygame
  V Load each
  V PredictorNetwork
    - Overfitting
  
  - Jan 3rd
  - ControllerNetwork
  - CircularEnvironment
  - prev_input -> prev_feature with retain_graph=True

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