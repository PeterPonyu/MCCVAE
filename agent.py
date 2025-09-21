from .environment import Env
from .utils import quiver_autoscale, l2_norm
import scanpy as sc
from anndata import AnnData
import torch
import tqdm
from typing import Optional, Literal
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors


class agent(Env):
    def __init__(
        self,
        adata: AnnData,
        layer: str = "counts",
        percent: float = 0.01,
        recon: float = 1.0,
        irecon: float = 0.0,
        beta: float = 1.0,
        dip: float = 0.0,
        tc: float = 0.0,
        info: float = 0.0,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 2,
        use_ode: bool = False,
        use_moco: bool = False,
        loss_mode: Literal["mse", "nb", "zinb"] = "nb",
        lr: float = 1e-4,
        vae_reg: float = 0.5,
        ode_reg: float = 0.5,
        moco_weight: float = 1,
        moco_T: float = 0.2,
        aug_prob: float = 0.5,
        mask_prob: float = 0.2,
        noise_prob: float = 0.7,
        use_qm: bool = True,
        device: torch.device = torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    ):
        super().__init__(
            adata=adata,
            layer=layer,
            percent=percent,
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
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
            aug_prob=aug_prob,
            mask_prob=mask_prob,
            noise_prob=noise_prob,
            use_qm=use_qm,
            device=device,
        )

    def fit(self, epochs: int = 1000):
        with tqdm.tqdm(total=int(epochs), desc="Fitting", ncols=150) as pbar:
            for i in range(int(epochs)):
                data_x, data_q, data_k = self.load_data()
                if self.use_moco:
                    self.step(data_x, data_q, data_k)
                else:
                    self.step(data_x)
                if (i + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "Loss": f"{self.loss[-1][0]:.2f}",
                            "ARI": f"{(self.score[-1][0]):.2f}",
                            "NMI": f"{(self.score[-1][1]):.2f}",
                            "ASW": f"{(self.score[-1][2]):.2f}",
                            "C_H": f"{(self.score[-1][3]):.2f}",
                            "D_B": f"{(self.score[-1][4]):.2f}",
                            "P_C": f"{(self.score[-1][5]):.2f}",
                        }
                    )
                pbar.update(1)
        return self

    def get_iembed(
        self,
    ):
        iembed = self.take_iembed(self.X)
        return iembed

    def get_latent(
        self,
    ):
        latent = self.take_latent(self.X)
        return latent

    def get_time(
        self,
    ):
        time = self.take_time(self.X)
        return time

    def get_impute(
        self, top_k: int = 30, alpha: float = 0.9, steps: int = 3, decay: float = 0.99
    ):
        T = self.take_transition(self.X, top_k)

        def multi_step_impute(T, X, steps, decay):
            X_current = X.copy()
            X_imputed = X.copy()
            for i in range(steps):
                X_current = T @ X_current
                X_imputed = X_imputed + decay**i * X_current
            X_imputed = X_imputed / (1 + sum(decay**i for i in range(steps)))
            return X_imputed

        def balanced_impute(T, X, alpha=0.5, steps=3, decay=0.9):
            X_imputed = multi_step_impute(T, X, steps, decay)
            X_balanced = (1 - alpha) * X + alpha * X_imputed
            return X_balanced

        return balanced_impute(T, self.X, alpha, steps, decay)

    def get_vfres(
        self,
        adata: AnnData,
        zs_key: str,
        E_key: str,
        vf_key: str = "X_vf",
        T_key: str = "cosine_similarity",
        dv_key: str = "X_dv",
        reverse: bool = False,
        run_neigh: bool = True,
        use_rep_neigh: Optional[str] = None,
        t_key: Optional[str] = None,
        n_neigh: int = 20,
        var_stabilize_transform: bool = False,
        scale: int = 10,
        self_transition: bool = False,
        smooth: float = 0.5,
        stream: bool = True,
        density: float = 1.0,
    ):
        grads = self.take_grad(self.X)
        adata.obsm[vf_key] = grads
        adata.obsp[T_key] = self.get_similarity(
            adata,
            zs_key=zs_key,
            vf_key=vf_key,
            reverse=reverse,
            run_neigh=run_neigh,
            use_rep_neigh=use_rep_neigh,
            t_key=t_key,
            n_neigh=n_neigh,
            var_stabilize_transform=var_stabilize_transform,
        )
        adata.obsm[dv_key] = self.get_vf(
            adata,
            T_key=T_key,
            E_key=E_key,
            scale=scale,
            self_transition=self_transition,
        )
        E = np.array(adata.obsm[E_key])
        V = adata.obsm[dv_key]
        E_grid, V_grid = self.get_vfgrid(
            E=E,
            V=V,
            smooth=smooth,
            stream=stream,
            density=density,
        )
        return E_grid, V_grid

    def get_similarity(
        self,
        adata: AnnData,
        zs_key: str,
        vf_key: str = "X_vf",
        reverse: bool = False,
        run_neigh: bool = True,
        use_rep_neigh: Optional[str] = None,
        t_key: Optional[str] = None,
        n_neigh: int = 20,
        var_stabilize_transform: bool = False,
    ):
        Z = np.array(adata.obsm[zs_key])
        V = np.array(adata.obsm[vf_key])
        if reverse:
            V = -V
        if var_stabilize_transform:
            V = np.sqrt(np.abs(V)) * np.sign(V)

        ncells = adata.n_obs

        if run_neigh or ("neighbors" not in adata.uns):
            if use_rep_neigh is None:
                use_rep_neigh = zs_key
            else:
                if use_rep_neigh not in adata.obsm:
                    raise KeyError(
                        f"`{use_rep_neigh}` not found in `.obsm` of the AnnData. Please provide valid `use_rep_neigh` for neighbor detection."
                    )
            sc.pp.neighbors(adata, use_rep=use_rep_neigh, n_neighbors=n_neigh)
        n_neigh = adata.uns["neighbors"]["params"]["n_neighbors"] - 1

        if t_key is not None:
            if t_key not in adata.obs:
                raise KeyError(
                    f"`{t_key}` not found in `.obs` of the AnnData. Please provide valid `t_key` for estimated pseudotime."
                )
            ts = adata.obs[t_key].values
            indices_matrix2 = np.zeros((ncells, n_neigh), dtype=int)
            for i in range(ncells):
                idx = np.abs(ts - ts[i]).argsort()[: (n_neigh + 1)]
                idx = np.setdiff1d(idx, i) if i in idx else idx[:-1]
                indices_matrix2[i] = idx

        vals, rows, cols = [], [], []
        for i in range(ncells):
            idx = adata.obsp["distances"][i].indices
            idx2 = adata.obsp["distances"][idx].indices
            idx2 = np.setdiff1d(idx2, i)
            idx = (
                np.unique(np.concatenate([idx, idx2]))
                if t_key is None
                else np.unique(np.concatenate([idx, idx2, indices_matrix2[i]]))
            )
            dZ = Z[idx] - Z[i, None]
            if var_stabilize_transform:
                dZ = np.sqrt(np.abs(dZ)) * np.sign(dZ)
            cos_sim = np.einsum("ij, j", dZ, V[i]) / (
                l2_norm(dZ, axis=1) * l2_norm(V[i])
            )
            cos_sim[np.isnan(cos_sim)] = 0
            vals.extend(cos_sim)
            rows.extend(np.repeat(i, len(idx)))
            cols.extend(idx)

        res = coo_matrix((vals, (rows, cols)), shape=(ncells, ncells))
        res.data = np.clip(res.data, -1, 1)
        return res.tocsr()

    def get_vf(
        self,
        adata: AnnData,
        T_key: str,
        E_key: str,
        scale: int = 10,
        self_transition: bool = False,
    ):
        T = adata.obsp[T_key].copy()

        if self_transition:
            max_t = T.max(1).A.flatten()
            ub = np.percentile(max_t, 98)
            self_t = np.clip(ub - max_t, 0, 1)
            T.setdiag(self_t)

        T = T.sign().multiply(np.expm1(abs(T * scale)))
        T = T.multiply(csr_matrix(1.0 / abs(T).sum(1)))
        if self_transition:
            T.setdiag(0)
            T.eliminate_zeros()

        E = np.array(adata.obsm[E_key])
        V = np.zeros(E.shape)

        for i in range(adata.n_obs):
            idx = T[i].indices
            dE = E[idx] - E[i, None]
            dE /= l2_norm(dE)[:, None]
            dE[np.isnan(dE)] = 0
            prob = T[i].data
            V[i] = prob.dot(dE) - prob.mean() * dE.sum(0)

        V /= 3 * quiver_autoscale(E, V)
        return V

    def get_vfgrid(
        self,
        E: np.ndarray,
        V: np.ndarray,
        smooth: float = 0.5,
        stream: bool = True,
        density: float = 1.0,
    ):
        grs = []
        for i in range(E.shape[1]):
            m, M = np.min(E[:, i]), np.max(E[:, i])
            diff = M - m
            m = m - 0.01 * diff
            M = M + 0.01 * diff
            gr = np.linspace(m, M, int(50 * density))
            grs.append(gr)

        meshes = np.meshgrid(*grs)
        E_grid = np.vstack([i.flat for i in meshes]).T

        n_neigh = int(E.shape[0] / 50)
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1)
        nn.fit(E)
        dists, neighs = nn.kneighbors(E_grid)

        scale = np.mean([g[1] - g[0] for g in grs]) * smooth
        weight = norm.pdf(x=dists, scale=scale)
        weight_sum = weight.sum(1)

        V_grid = (V[neighs] * weight[:, :, None]).sum(1)
        V_grid /= np.maximum(1, weight_sum)[:, None]

        if stream:
            E_grid = np.stack(grs)
            ns = E_grid.shape[1]
            V_grid = V_grid.T.reshape(2, ns, ns)

            mass = np.sqrt((V_grid * V_grid).sum(0))
            min_mass = 1e-5
            min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
            cutoff1 = mass < min_mass

            length = np.sum(np.mean(np.abs(V[neighs]), axis=1), axis=1).reshape(ns, ns)
            cutoff2 = length < np.percentile(length, 5)

            cutoff = cutoff1 | cutoff2
            V_grid[0][cutoff] = np.nan
        else:
            min_weight = np.percentile(weight_sum, 99) * 0.01
            E_grid, V_grid = (
                E_grid[weight_sum > min_weight],
                V_grid[weight_sum > min_weight],
            )
            V_grid /= 3 * quiver_autoscale(E_grid, V_grid)

        return E_grid, V_grid
