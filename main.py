from environments import LinearSpectrumEnvironment
from trainer import Trainer

if __name__ == '__main__':
    env = LinearSpectrumEnvironment()
    trainer = Trainer(env)
    trainer.train()