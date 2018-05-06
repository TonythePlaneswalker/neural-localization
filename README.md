# Active Neural Localization in Noisy Environments

## Installation

First, install OpenAI [Gym](https://github.com/openai/gym#installation), [PyTorch](https://pytorch.org) and [Visdom](https://github.com/facebookresearch/visdom).

Then, install the gridworld environment by running
```bash
pip install -e gridworld
```

To train an agent, first open visdom by running
```bash
python -m visdom.server
```

Then start training by running
```bash
python a2c_v2.py
```
and navigate to `http://localhost:8097` to view the training plots.

To test the trained agent, run
```bash
python test.py --model [path_to_saved_model]
```

Run `python a2c_v2.py -h` or `python test.py -h` to see more options. 

You can also play with the environment yourself by running `python play.py`. Have fun!
