import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchsde
import torchcde
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.nn.utils import spectral_norm

#-- Lipschitz-constrained activation ---
class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            
            model.append(LipSwish())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


class GeneratorFunc(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'general'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)

        # learnable scalar scales
        self.scale_f = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.scale_g = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)

        # raw outputs
        raw_f = self._drift(tx)                       # (B, hidden)
        raw_g = self._diffusion(tx)                   # (B, hidden*noise)

        # reshape diffusion to (B,hidden,noise)
        raw_g = raw_g.view(x.size(0), self._hidden_size, self._noise_size)

        f =  self.scale_f * raw_f
        g =  self.scale_g * raw_g
        #g = torch.clamp(g, -5.0, 5.0)
        #print("→ raw_g std:", raw_g.std().item(), "| clamped_g std:", g.std().item())
        return f, g


class Generator(torch.nn.Module):
    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, data_size, bias=True)

    def forward(self, ts, batch_size):
        
        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        x0 = self._initial(init_noise)
        x0 += 0.05 * torch.randn_like(x0) 
        print("→ x0   mean/std:", x0.mean().item(), x0.std().item())

        delta_t = 0.2

        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='euler', dt=delta_t)
        xs = xs.transpose(0, 1)
        ys = self._readout(xs)
        # w_std = self._readout.weight.std().item()
        # print(f"→ readout weight std={w_std:.3f}")
        
        # dys = ys[:,1:] - ys[:,:-1]           
        # print("→ ys per‐step std:", dys.std().item(),
        # " overall ys std:", ys.std().item())
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))



class DiscriminatorFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)



class Discriminator(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()

        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size*2, 1)


    def forward(self, ys_coeffs):
        Y = torchcde.LinearInterpolation(ys_coeffs)
        
        delta_t = 0.5

        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)  
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, method='rk4', backend='torchdiffeq', adjoint=False, options={'step_size': delta_t})
        
        hs_mean = hs.mean(dim=1)
        hs_std = hs.std(dim=1)
        score = self._readout(torch.cat([hs_mean, hs_std], dim=1))
        return score.squeeze(-1)  # (batch_size,)


# ---- Split and scale dei log-prices ----
def split_and_scale(log_prices):
    if isinstance(log_prices, pd.Series):
        log_prices = log_prices.values.reshape(-1, 1)

    train_prices, test_prices = train_test_split(log_prices, test_size=0.25, shuffle=False)

    train_prices = train_prices.reshape(-1, 1)
    test_prices = test_prices.reshape(-1, 1)

    scaler = RobustScaler()
    train_prices_scaled = scaler.fit_transform(train_prices)
    test_prices_scaled = scaler.transform(test_prices)

    return train_prices_scaled, test_prices_scaled, scaler


class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len=50):
        self.seq_len = seq_len
        self.data = series
        self.ts = torch.from_numpy(np.arange(0, seq_len)).float()  

    def __len__(self):
        return len(self.data) - self.seq_len +1

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.seq_len]  
        y = torch.from_numpy(window.astype(np.float32))
        return self.ts, y            
    

import torch
from torch.autograd import grad

def gradient_penalty(D, real, fake, λ=10.0):
    """
    D    : discriminator returning per-sample scores of shape (B,)
    real : tensor (B, T, C)
    fake : tensor (B, T, C)
    """
    B = real.size(0)
    device = real.device

    # Interpolate between real and fake
    ε = torch.rand(B, 1, 1, device=device)
    x̂ = ε * real + (1 - ε) * fake
    x̂.requires_grad_(True)

    assert x̂.requires_grad, "Gradient penalty will silently break – x̂ is not requiring grad"

    # Forward pass
    scores = D(x̂)                               # (B,)
    assert scores.requires_grad, "Discriminator output does not depend on interpolated input!"

    # Compute gradients ∇_x̂ D(x̂)
    grads = grad(
        outputs = scores,
        inputs  = x̂,
        grad_outputs=torch.ones_like(scores, device=device),
        create_graph=True,
        retain_graph=True
    )[0]                                         # (B, T, C)

    grads_x = grads[..., 1:]  

    # Per-sample norm
    grads_x = grads_x.reshape(B, -1)                 # (B, T*C)
    grad_norm = grads_x.norm(2, dim=1)             # (B,)

    # Penalty term
    penalty = λ * ((grad_norm - 1) ** 2).mean()
    
   
    with torch.no_grad():
        deviation = (grad_norm - 1).abs().mean()
        print(f"[Debug] Mean |∥∇D/∂x∥ - 1| = {deviation.item():.4f}")
    return penalty
