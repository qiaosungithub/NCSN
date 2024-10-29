import flax.nnx as nn
import jax
import jax.numpy as jnp
import sys
import json
import optax
sys.path.append('..')
# from models import ResNet50, _ResNet1

# model = _ResNet1(num_classes=1000,rngs=nn.Rngs(0),dtype=jnp.float32)

def display_params(model: nn.Module) -> dict:
    _, state = nn.split(model)
    tree = state.to_pure_dict()
    d = jax.tree_map(lambda x:f'Tensor{x.shape}' if isinstance(x, jnp.ndarray) else x, tree )
    return d

def display_model(model: nn.Module) -> dict:
    if not isinstance(model, nn.Module): return str(model)
    attrs = {}
    arrays = {}
    modules = {}
    for name, value in vars(model).items():
        if name.startswith('_'):
            continue
        if not type(value) in [int, float, str, bool, type(None)]:
            # print(name, type(value))
            if 'flax.nnx.variablelib' in str(type(value)):
                arrays[name] = f'Tensor{value.shape}'
            if isinstance(value, nn.Module):
                modules[name] = display_model(value)
            if isinstance(value, list):
                modules[name] = {i:display_model(v) for i,v in enumerate(value)}
            continue
        attrs[name] = value
    # turn attrs to a string
    attrs = ', '.join([f'{k}={v}' for k,v in attrs.items()])
    to_display = [model.__class__.__name__ + f'({attrs})', arrays, modules]
    return tuple([c for c in to_display if c])

