from .model import CODEVAE
from .mixin import envMixin
import numpy as np
import torch
from sklearn.cluster import KMeans


class Env(CODEVAE, envMixin):
    def __init__(
        self,
        adata,
        layer,
        percent,
        recon,
        irecon,
        beta,
        dip,
        tc,
        info,
        hidden_dim,
        latent_dim,
        i_dim,
        use_ode,
        use_moco,
        loss_mode,
        lr,
        vae_reg,
        ode_reg,
        moco_weight,
        moco_T,
        aug_prob,
        mask_prob, 
        noise_prob,
        use_qm,
        device,
        *args,
        **kwargs,
    ):
        self._register_anndata(adata, layer, latent_dim)
        self.batch_size = int(percent * self.n_obs)
        self.aug_prob = aug_prob
        self.mask_prob = mask_prob
        self.noise_prob = noise_prob
        super().__init__(
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            state_dim=self.n_var,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            use_ode=use_ode,
            use_moco=use_moco,
            loss_mode=loss_mode,
            lr=lr,
            vae_reg=vae_reg,
            ode_reg=ode_reg,
            moco_weight=moco_weight,
            moco_T=moco_T,
            use_qm=use_qm,
            device=device,
        )
        self.score = []

    def load_data(
        self,
    ):
        if self.use_moco:
            data, data_q, data_k, idx = self._sample_data()
            self.idx = idx
            return data, data_q, data_k
        else:
            data, idx = self._sample_data()
            self.idx = idx
            return data, None, None

    def step(self, *data):
        if self.use_moco:
            self.update(data[0], data[1], data[2])
            latent = self.take_latent(data[0])
        else:
            self.update(data[0])
            latent = self.take_latent(data[0])
        score = self._calc_score(latent)
        self.score.append(score)

    def _sample_data(
        self,
    ):
        idx = np.random.permutation(self.n_obs)
        idx_ = np.random.choice(idx, self.batch_size)
        data = self.X[idx_, :]
        if self.use_moco:
            return data, self._augment(data), self._augment(data), idx_
        else:
            return data, idx_

    def _augment(self, profile):
        if isinstance(profile, torch.Tensor):
            profile_np = profile.cpu().numpy()
        else:
            profile_np = profile.copy()

        profile_np = profile_np.astype(np.float32)

        if np.random.rand() < self.aug_prob:
            # Masking
            mask = np.random.choice(
                [True, False], self.n_var, p=[self.mask_prob, 1 - self.mask_prob]
            )
            profile_np[:, mask] = 0
            # Gaussian Noise
            mask = np.random.choice(
                [True, False], self.n_var, p=[self.noise_prob, 1 - self.noise_prob]
            )
            noise = np.random.normal(0, 0.2, (profile_np.shape[0], np.sum(mask)))
            profile_np[:, mask] += noise

        # Clip values to be non-negative to ensure log1p stability
        profile_np = np.clip(profile_np, 0, None)

        if isinstance(profile, torch.Tensor):
            return torch.from_numpy(profile_np).to(profile.device)
        else:
            return profile_np

    def _register_anndata(self, adata, layer: str, latent_dim):
        self.X = adata.layers[layer].toarray()
        self.n_obs = adata.shape[0]
        self.n_var = adata.shape[1]
        self.labels = KMeans(latent_dim).fit_predict(self.X)
        return
