import flax.nnx as nn
from jax import random

rngs = nn.Rngs(0)
print(random.randint(rngs(), (10,), 0, 10))