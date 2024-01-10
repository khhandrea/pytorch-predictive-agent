from environments import LinearSpectrumEnvironment
from trainer import Trainer

if __name__ == '__main__':
    env = LinearSpectrumEnvironment(render_mode='human')
    trainer = Trainer(env)
    trainer.train()