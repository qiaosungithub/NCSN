try:
    from NCSN.utils import *
    from NCSN.model import *
    from NCSN.langevin import *
except:
    from utils import *
    from model import *
    from langevin import *
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import os

def get_outdir(time_str):
    outdir = f"NCSN/log/{time_str}.out"
    return outdir

def get_sample_dir(time_str):
    dir = f"/nobackup/users/sqa24/NCSN/{time_str}/samples"
    mkdir(dir)
    return dir

def get_denoising_dir(time_str, epoch):
    dir = f"/nobackup/users/sqa24/NCSN/{time_str}/denoising/{(epoch + 1):03d}"
    mkdir(dir)
    return dir

def get_model_path(time_str, epoch):
    return f"/nobackup/users/sqa24/NCSN/{time_str}/models/{epoch:03d}.pth"

def save_py_files(time_str):
    # copy all .py files in "NCSN" to "nobackup/users/sqa24/NCSN/models/time_str"
    mkdir(f"/nobackup/users/sqa24/NCSN/{time_str}")
    mkdir(f"/nobackup/users/sqa24/NCSN/{time_str}/models")
    mkdir(f"/nobackup/users/sqa24/NCSN/{time_str}/codes")
    os.system(f"cp -r NCSN/*.py /nobackup/users/sqa24/NCSN/{time_str}/codes/")
    print("code files copied")

def cal_noise_level(init, final, steps):
    sigmas = []
    init = np.log(init)
    final = np.log(final)
    for i in range(steps):
        sigmas.append(np.exp(init + (final - init) * i / (steps - 1)))
    # print("noise levels: ", sigmas)
    return sigmas

def save_image(images, filename):
    images = images.clamp(0, 1)
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2, pad_value=1)
    torchvision.utils.save_image(grid, filename)

def should_eval(epoch, eval_freq):
    if eval_freq == "square":
        return is_square(epoch)
    return (epoch+1) % eval_freq == 0

def train(epochs, model, optimizer, criterion, train_loader, val_loader, sigmas, eps, T, time_str, eval_freq=5):
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    mkdir(f"NCSN/log")

    save_py_files(time_str)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.cuda()

    outdir = get_outdir(time_str)
    sample_dir = get_sample_dir(time_str)

    with open(outdir, 'a') as f:
        f.write(str(model) + '\n')
        f.write(str(optimizer) + '\n')
        f.write("sigmas: " + str(sigmas) + '\n')
        f.write("eps: " + str(eps) + '\n')
        f.write("T: " + str(T) + '\n')

    best_matching_loss = float('inf')
    train_losses = []
    # sigmas = cal_noise_level(init_sigma, final_sigma, n_sigma)
    n_sigma = len(sigmas)
    sigmas_t = torch.tensor(sigmas, device=device)

    best_mse = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for b in progress_bar:
            images, _ = b
            sigma_indices = torch.randint(0, n_sigma, (images.size(0),), device=device)
            sigma_batch = sigmas_t[sigma_indices].view(-1, 1, 1, 1)

            # add noise to images
            images = images.cuda()
            noise = torch.randn_like(images) * sigma_batch
            images_noise = images + noise

            optimizer.zero_grad()
            outputs = model(images_noise, sigma_indices)
            outputs = outputs * sigma_batch
            target = - noise / sigma_batch
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'Train Loss': running_loss / (progress_bar.n + 1)})

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        print("Epoch {i}: train loss {loss}".format(i=epoch, loss=train_loss))
        # save it to outdir (which is a file)
        with open(outdir, 'a') as f:
            f.write("Epoch {i}: train loss {loss}\n".format(i=epoch, loss=train_loss))

        # save model
        if train_loss < best_matching_loss:
            best_matching_loss = train_loss
        model_path = get_model_path(time_str, epoch)
        torch.save(model.state_dict(), model_path)

        if should_eval(epoch, eval_freq):
            # generate samples
            model.eval()
            with torch.no_grad():
                x = torch.rand(64, 1, 28, 28).cuda()
                samples = langevin(model, x, sigmas, eps, T, clamp=False)
                save_image(samples, sample_dir + "/{i:03d}.png".format(i=epoch))

            # evaluate denoising
            c_mse, r_mse, original, broken, recovered = evaluate_denoising(model, sigmas, eps, T, val_loader, outdir)
            denoising_dir = get_denoising_dir(time_str, epoch)
            torchvision.utils.save_image(
                original, denoising_dir + "/groundtruth.png", nrow=10)
            torchvision.utils.save_image(
                broken, denoising_dir + "/corrupted.png", nrow=10)
            torchvision.utils.save_image(
                recovered, denoising_dir + "/recovered.png", nrow=10)
            if r_mse < best_mse:
                print(f'Current best MSE: {best_mse} -> {r_mse}')
                with open(outdir, 'a') as f:
                    f.write(f'Current best MSE: {best_mse} -> {r_mse}\n')
                best_mse = r_mse

@torch.no_grad()
def make_dataset(model, sigmas, eps, T, n_samples=1000, save=True, save_dir='./NCSN/generated/'):
    model.eval()
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    x = torch.rand(n_samples, 1, 28, 28).cuda()
    samples = langevin(model, x, sigmas, eps, T, clamp=False)

    mkdir(save_dir)
    
    if save:
        for j in tqdm(range(n_samples)): 
            torchvision.utils.save_image(samples[j], os.path.join(save_dir, "{:>03d}.png".format(j)))

def evaluate_denoising(score_model, sigmas, eps, T, val_loader, outdir):
    mse = corruption_mse = 0
    n_batches = 0
    score_model.eval()

    # pbar = tqdm(total=len(val_loader.dataset))
    pbar = tqdm(total=2000)
    pbar.set_description('Eval')
    for data, _ in val_loader:
        # here we only evaluate four batches (2000 images)
        n_batches += data.shape[0]
        data = data.cuda()
        broken_data, mask = corruption(data, type_='ebm')

        recovered_img = langevin_masked(score_model, broken_data, sigmas, eps, T, mask, save=False, clamp=False)

        mse += np.mean((data.detach().cpu().numpy().reshape(-1, 28 * 28) - recovered_img.detach().cpu().numpy().reshape(-1, 28 * 28)) ** 2, -1).sum().item()
        corruption_mse += np.mean((data.detach().cpu().numpy().reshape(-1, 28 * 28) - broken_data.detach().cpu().numpy().reshape(-1, 28 * 28)) ** 2, -1).sum().item()

        pbar.update(data.shape[0])
        pbar.set_description('Corruption MSE: {:.6f}, Recovered MSE: {:.6f}'.format(corruption_mse / n_batches, mse / n_batches))
        with open(outdir, 'a') as f:
            f.write('Corruption MSE: {:.6f}, Recovered MSE: {:.6f}\n'.format(corruption_mse / n_batches, mse / n_batches))
        # show the image
        # import matplotlib.pyplot as plt
        # plt.imshow(recovered_img[0].detach().cpu().numpy().reshape(28, 28), cmap='gray')
        # plt.show()
        # break
        if n_batches >= 2000:
            break

    pbar.close()
    return (corruption_mse / n_batches, mse / n_batches, data[:100].detach().cpu(), broken_data[:100].detach().cpu(), recovered_img[:100].detach().cpu())
