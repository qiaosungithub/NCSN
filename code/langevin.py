import jax
import os
from math import sqrt
import jax.numpy as jnp
from utils.utils import save_img

# TODO

def apply_langevin(state, x, alpha, noise, indices, mask=None):
    grad, _, _ = state.apply_fn(state.graphdef, state.params, state.rng_states, state.batch_stats, state.useless_variable_state, False, x, indices)
    if mask is not None:
        x = x + (alpha / 2 * grad + sqrt(alpha) * noise) * mask
    else:
        x = x + alpha / 2 * grad + sqrt(alpha) * noise
    return x, grad

fast_apply_langevin = jax.pmap(
    apply_langevin,
    axis_name='batch',
)

def langevin(state, shape, sigmas, eps, T, rngs, whole_process=False, clamp=False, verbose=False):
    """
    rngs: a Rng class instance
    """

    # it's better not to clamp
    bs = shape[0]
    x = jax.random.normal(rngs.evaluation()+jax.process_index(), shape=shape)
    if whole_process:
        assert bs <= 20, "batch size should be less than 20 if you want to save the whole process"
        all_samples = []
    
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        alpha = eps * (sigma ** 2) / (sigmas[-1] ** 2)
        indices = i * jnp.ones(bs, dtype=jnp.int32)
        for t in range(T):
            noise = jax.random.normal(rngs.evaluation()+jax.process_index(), shape=x.shape)
            assert indices.shape == ([bs,])
            x, grad = fast_apply_langevin(state, x, alpha, noise, indices)
            if clamp:
                x = jnp.clip(x, 0, 1)
            if verbose:
                grad_norm = jnp.linalg.norm(grad.view(bs, -1), dim=1).mean()
                image_norm = jnp.linalg.norm(x.view(bs, -1), dim=1).mean()
                noise_norm = jnp.linalg.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                snr = jnp.sqrt(alpha) * grad_norm / noise_norm # signal to noise ratio
                grad_mean_norm = jnp.linalg.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
                if jax.process_index() == 0:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                                    i, alpha, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
        if whole_process:
            all_samples.append(x)

    if whole_process:
        all_samples = jnp.stack(all_samples, axis=0)
        all_samples = all_samples.reshape(-1, *shape[2:])
        return x, all_samples
    else:
        return x

    if save:
        assert x.shape[0] == 10
        # assert len(sigmas) == 10
        assert epochs is not None
        if time_str is not None:
            save_dir = f'./NCSN/denoising_process/{time_str}/'
        else:
            save_dir = './NCSN/denoising_process/'
        filename = '{:>03d}.png'.format(epochs)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # concatenate all samples
        all_samples = all_samples[:10]
        all_samples = torch.cat(all_samples, dim=0)
        # print(all_samples.shape)
        assert all_samples.shape == torch.Size([100, 1, 28, 28])
        # save the image
        grid = torchvision.utils.make_grid(all_samples, nrow=10, padding=2, pad_value=1)
        torchvision.utils.save_image(grid, os.path.join(save_dir, filename))

    return x

def langevin_masked(state, x, sigmas, eps, T, rngs, mask, whole_process=False, clamp=False, verbose=False):
    """
    rngs: a Rng class instance
    """

    # it's better not to clamp
    bs = x.shape[0]
    if whole_process:
        assert bs <= 20, "batch size should be less than 20 if you want to save the whole process"
        all_samples = []
    
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        alpha = eps * (sigma ** 2) / (sigmas[-1] ** 2)
        indices = i * jnp.ones(bs, dtype=jnp.int32)
        for t in range(T):
            noise = jax.random.normal(rngs.evaluation()+jax.process_index(), shape=x.shape)
            assert indices.shape == ([bs,])
            x, grad = fast_apply_langevin(state, x, alpha, noise, indices, mask=mask)
            if clamp:
                x = jnp.clip(x, 0, 1)
            if verbose:
                grad_norm = jnp.linalg.norm(grad.view(bs, -1), dim=1).mean()
                image_norm = jnp.linalg.norm(x.view(bs, -1), dim=1).mean()
                noise_norm = jnp.linalg.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                snr = jnp.sqrt(alpha) * grad_norm / noise_norm # signal to noise ratio
                grad_mean_norm = jnp.linalg.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2
                if jax.process_index() == 0:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                                    i, alpha, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))
        if whole_process:
            all_samples.append(x)

    if whole_process:
        return x, all_samples
    else:
        return x