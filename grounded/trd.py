"""Transition Realism Discriminator (TRD).

Learns to distinguish real transitions from world-model-imagined
transitions in latent space. Outputs a trust score in (0, 1).
"""

import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax.nets as nn

f32 = jnp.float32


class SpectralNormLinear(nj.Module):
    """Linear layer with spectral normalization via power iteration.

    Constrains the Lipschitz constant of the layer by normalizing
    the weight matrix by its largest singular value estimate.
    """

    units: int = 256
    n_power_iter: int = 1

    def __call__(self, x):
        in_dim = x.shape[-1]
        kernel = self.value(
            'kernel', nn.init('normal_in'), (in_dim, self.units))
        u = self.value('u', lambda shape: jax.random.normal(
            nj.seed(), shape), (1, self.units))
        # Power iteration to estimate sigma_max
        u_hat, v_hat = u, None
        for _ in range(self.n_power_iter):
            v_hat = u_hat @ kernel.T  # (1, in_dim)
            v_hat = v_hat / (jnp.linalg.norm(v_hat, axis=-1, keepdims=True) + 1e-8)
            u_hat = v_hat @ kernel  # (1, units)
            u_hat = u_hat / (jnp.linalg.norm(u_hat, axis=-1, keepdims=True) + 1e-8)
        # Persist updated u via ninjax write (not value, which is read-only after init)
        self.write('u', u_hat)
        sigma = (v_hat @ kernel @ u_hat.T).squeeze()  # scalar
        kernel_sn = kernel / (sigma + 1e-8)
        bias = self.value('bias', jnp.zeros, (self.units,))
        return x @ kernel_sn.astype(x.dtype) + bias.astype(x.dtype)


class TRD(nj.Module):
    """Transition Realism Discriminator.

    Given a transition (z_t, a_t, z_{t+1}) in latent space,
    outputs a trust score indicating how "real" the transition looks.

    Uses Spectral Normalization to prevent the discriminator from
    becoming too powerful (classic GAN stability technique).
    """

    hidden: int = 256

    def __call__(self, z_t, a_t, z_next):
        """Score a transition's realism.

        Args:
            z_t: Current latent state. # (B, z_dim)
            a_t: Action. # (B, a_dim)
            z_next: Next latent state. # (B, z_dim)

        Returns:
            Trust score in (0, 1). # (B,)
        """
        x = jnp.concatenate([z_t, a_t, z_next], axis=-1)  # (B, z_dim*2 + a_dim)
        x = self.sub('sn0', SpectralNormLinear, units=self.hidden)(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = self.sub('sn1', SpectralNormLinear, units=self.hidden)(x)
        x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = self.sub('sn2', SpectralNormLinear, units=1)(x)
        return jax.nn.sigmoid(x.squeeze(-1))  # (B,)

    @staticmethod
    def train_loss(scores_real, scores_fake, label_smooth=0.9):
        """Binary cross-entropy loss with label smoothing.

        Args:
            scores_real: TRD output for real transitions. # (B,)
            scores_fake: TRD output for imagined transitions. # (B,)
            label_smooth: Smoothed label for real samples (default 0.9).

        Returns:
            Scalar loss.
        """
        eps = 1e-7
        # Real: target = label_smooth, full BCE
        loss_real = -(label_smooth * jnp.log(scores_real + eps) +
                      (1 - label_smooth) * jnp.log(1 - scores_real + eps))
        # Fake: target = 0
        loss_fake = -jnp.log(1 - scores_fake + eps)
        return (loss_real + loss_fake).mean()
