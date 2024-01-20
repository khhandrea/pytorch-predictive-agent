# Python-predict-agent

## TODO
  v 1d feature, discrete action FeatureExtractorInverseNetwork
  v average loss
  v Save
  v Load
  - GPU
  - pygame
  
  - Jan 3rd
  - PredictorNetwork
  - ControllerNetwork
  - CircularEnvironment

  - Jan 4th
  - ComplexEnvironment
  - 2d FeatureExtractorInverseNetwork
  - Continuous action

  - Others
  - Name to predictive-agent
  - Arguments convention (white space)
  - Best checkpoint
  - Fine tuning
  - Makefile setting
    - ref. https://kimjingo.tistory.com/203
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