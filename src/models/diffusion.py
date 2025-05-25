from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from src.utils.setup import DEVICE

from .helpers import cosine_beta_schedule, extract

Sample = namedtuple("Sample", "sample chains")


class GaussianDiffusion(nn.Module):
    """
    Base Gaussian diffusion model
    """

    def __init__(self, model, data_dim=3, n_timesteps=20):
        super().__init__()

        self.model = model
        self.data_dim = data_dim

        betas = cosine_beta_schedule(n_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]]
        )  # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        self.n_timesteps = int(n_timesteps)

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )  # helper function to register buffer from float64 to float32

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """ """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.enable_grad()
    def p_mean_variance(self, x, t):
        t = t.clone().float().detach()

        if self.swg_guidance:
            x.requires_grad_(True)

            if x.grad is not None:
                x.grad.zero_()

            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

            t_long = t.clone().detach().long()

            epsilon = self.model(x=x, time=t)

            w_term = (
                x[:, -1]
                - extract(self.sqrt_one_minus_alphas_cumprod, t_long, x[:, -1].shape)
                * epsilon[:, -1]
            )
            w_term = torch.log(w_term)
            w_term_grad = torch.autograd.grad(w_term.sum(), x, retain_graph=False)[0]

            sqrt_one_minus_alphas_cumprod = extract(
                self.sqrt_one_minus_alphas_cumprod, t_long, x.shape
            )
            q_guidance_term = -sqrt_one_minus_alphas_cumprod * (w_term_grad)

            epsilon = epsilon + self.guidance_scale * q_guidance_term

        else:

            epsilon = self.model(x=x, time=t)

        t = t.detach().to(torch.int64)

        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.inference_mode()
    def p_sample(self, x, t):
        b, *_, device = *x.shape, x.device

        batched_time = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=batched_time)

        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        x_pred = model_mean + (0.5 * model_log_variance).exp() * noise

        return x_pred

    # @torch.inference_mode()
    def p_sample_loop(self, shape):
        """
        Classical DDPM (check this) sampling algorithm with repaint sampling

            Parameters:
                shape,

            Returns:
            sample

        """

        device = self.betas.device

        x = torch.randn(shape, device=device)

        chain = [x] if self.return_chain else None

        for t in tqdm(
            reversed(range(0, self.n_timesteps)),
            desc="sampling loop time step",
            total=self.n_timesteps,
            disable=self.disable_progess_bar,
        ):
            x = self.p_sample(x=x, t=t)
            if self.return_chain:
                chain.append(x)

        if self.return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)

    # @torch.inference_mode()
    def forward(self):
        """ """

        return self.p_sample_loop(shape=(self.batch_size, self.data_dim))

    def setup_sampling(
        self,
        batch_size=32,
        guidance_scale=1,
        disable_progess_bar=False,
        return_chain=False,
    ):
        self.batch_size = batch_size
        if guidance_scale > 0:
            self.swg_guidance = True
        else:
            self.swg_guidance = False
        self.guidance_scale = guidance_scale
        self.disable_progess_bar = disable_progess_bar
        self.return_chain = return_chain

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, t):

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        t = torch.tensor(t, dtype=torch.float, requires_grad=True)

        x_noisy.requires_grad = True
        noise.requires_grad = True

        pred_epsilon = self.model(x_noisy, t)

        assert noise.shape == pred_epsilon.shape

        loss = F.mse_loss(pred_epsilon, noise, reduction="none")

        weighted_loss = loss * self.loss_weights

        return weighted_loss.mean()

    def loss(self, x):

        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        return self.p_losses(x, t)


class GaussianDiffusion_Beta(nn.Module):
    """
    Gaussian diffusion model
    """

    def __init__(
        self, model, data_dim=3, n_timesteps=20, batch_size=1024, energy_weight=2
    ):
        super().__init__()

        self.model = model
        self.data_dim = data_dim

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.n_timesteps = int(n_timesteps)

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )  # helper function to register buffer from float64 to float32

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_weights = self.get_loss_weights(batch_size, energy_weight)

    def get_loss_weights(self, batch_size, energy_weight):
        loss_weights = torch.ones((batch_size, self.data_dim), device=DEVICE)
        loss_weights[:, -1] = energy_weight
        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """ """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.enable_grad()
    def p_mean_variance(self, x, t):

        batched_beta = torch.full(
            (self.batch_size,), self.beta, device=DEVICE, dtype=torch.float
        )

        if self.swg_guidance:
            x.requires_grad_(True)

            if x.grad is not None:
                x.grad.zero_()

            epsilon = self.model(x=x, time=t, beta=batched_beta)

            w_term = (
                x[:, -1]
                - extract(self.sqrt_one_minus_alphas_cumprod, t, x[:, -1].shape)
                * epsilon[:, -1]
            )

            w_term = torch.log(w_term)
            w_term_grad = torch.autograd.grad(w_term.sum(), x, retain_graph=False)[0]

            sqrt_one_minus_alphas_cumprod = extract(
                self.sqrt_one_minus_alphas_cumprod, t, x.shape
            )
            swg_guidance_term = -sqrt_one_minus_alphas_cumprod * (w_term_grad)

            epsilon = epsilon + self.guidance_scale * swg_guidance_term

        else:

            epsilon = self.model(x=x, time=t, beta=batched_beta)

        t = t.detach().to(torch.int64)

        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
            x_recon[:, -1].clamp_(0.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )

        return model_mean, posterior_variance, posterior_log_variance

    # @torch.inference_mode()
    def p_sample(self, x, t):
        b, *_, device = *x.shape, x.device

        batched_time = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=batched_time)

        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        x_pred = model_mean + (0.5 * model_log_variance).exp() * noise

        return x_pred

    # @torch.inference_mode()
    def p_sample_loop(self, shape):
        """
        Classical DDPM (check this) sampling algorithm with repaint sampling

            Parameters:
                shape

            Returns:
            sample

        """

        device = self.betas.device

        x = torch.randn(shape, device=device)

        chain = [x] if self.return_chain else None

        for t in tqdm(
            reversed(range(0, self.n_timesteps)),
            desc="sampling loop time step",
            total=self.n_timesteps,
            disable=self.disable_progess_bar,
        ):
            x = self.p_sample(x=x, t=t)
            if self.return_chain:
                chain.append(x)

        if self.return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, chain)

    def forward(self):

        return self.p_sample_loop(shape=(self.batch_size, self.data_dim))

    def setup_sampling(
        self,
        batch_size=32,
        beta=1,
        guidance_scale=1,
        clip_denoised=True,
        disable_progess_bar=False,
        return_chain=False,
    ):
        self.batch_size = batch_size
        self.beta = beta
        if guidance_scale > 0:
            self.swg_guidance = True
        else:
            self.swg_guidance = False
        self.guidance_scale = guidance_scale
        self.clip_denoised = clip_denoised
        self.disable_progess_bar = disable_progess_bar
        self.return_chain = return_chain

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, t, beta):

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        pred_epsilon = self.model(x_noisy, t, beta)

        assert noise.shape == pred_epsilon.shape

        loss = F.mse_loss(pred_epsilon, noise, reduction="none")

        weighted_loss = loss * self.loss_weights

        return weighted_loss.mean()

    def loss(self, x, beta):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()

        return self.p_losses(x, t, beta)
