
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE

class CODEVAE(scviMixin, dipMixin, betatcMixin, infoMixin):
    def __init__(
        self,
        recon,
        irecon,
        beta,
        dip,
        tc,
        info,
        state_dim,
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
        use_qm,
        moco_T,
        device,
        *args,
        **kwargs,
    ):
        self.use_ode = use_ode
        self.use_moco = use_moco
        self.use_qm = use_qm
        self.loss_mode = loss_mode
        self.recon = recon
        self.irecon = irecon
        self.beta = beta
        self.dip = dip
        self.tc = tc
        self.info = info
        self.moco_weight = moco_weight
        self.nn = VAE(
            state_dim,
            hidden_dim,
            latent_dim,
            i_dim,
            use_ode,
            use_moco,
            loss_mode,
            moco_T,
            device,
        )
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.vae_reg = vae_reg
        self.ode_reg = ode_reg
        self.device = device
        self.loss = []

        # Initialize MoCo loss criterion
        self.moco_criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def take_latent(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        if self.use_ode:
            outputs = self.nn.encoder(state)
            q_z, q_m, q_s, t = outputs
            t = t.cpu()
            t_sorted, sort_idx, sort_idxr = np.unique(
                t, return_index=True, return_inverse=True
            )
            t_sorted = torch.tensor(t_sorted)
            if self.use_qm:
                q_m_sorted = q_m[sort_idx]
                z0 = q_m_sorted[0]
                q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
                q_z_ode = q_z_ode[sort_idxr]
                return (self.vae_reg * q_m + self.ode_reg * q_z_ode).cpu().numpy()
            else:
                q_z_sorted = q_z[sort_idx]
                z0 = q_z_sorted[0]
                q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
                q_z_ode = q_z_ode[sort_idxr]
                return (self.vae_reg * q_z + self.ode_reg * q_z_ode).cpu().numpy()
        else:
            outputs = self.nn.encoder(state)
            q_z, q_m, q_s = outputs
            return q_z.cpu().numpy() if not self.use_qm else q_m.cpu().numpy()

    @torch.no_grad()
    def take_iembed(self, state):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        if self.use_ode:
            outputs = self.nn.encoder(states)
            q_z, q_m, q_s, t = outputs
            t = t.cpu()
            t_sorted, sort_idx, sort_idxr = np.unique(
                t, return_index=True, return_inverse=True
            )
            t_sorted = torch.tensor(t_sorted)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_idxr]

            le = self.nn.latent_encoder(q_z)
            le_ode = self.nn.latent_encoder(q_z_ode)
            return (self.vae_reg * le + self.ode_reg * le_ode).cpu().numpy()
        else:
            # The forward pass of the VAE handles the moco case
            outputs = self.nn(states)
            if self.use_moco:
                if self.loss_mode == "zinb":
                    q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl, logits, labels = outputs
                else:
                    q_z, q_m, q_s, pred_x, le, pred_xl, logits, labels = outputs
            else:
                if self.loss_mode == "zinb":
                    q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl = outputs
                else:
                    q_z, q_m, q_s, pred_x, le, pred_xl = outputs
            return le.cpu().numpy()

    @torch.no_grad()
    def take_time(self, state):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        outputs = self.nn.encoder(states)
        _, _, _, t = outputs
        return t.detach().cpu().numpy()

    @torch.no_grad()
    def take_grad(self, state):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        outputs = self.nn.encoder(states)
        q_z, q_m, q_s, t = outputs
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        return grads

    @torch.no_grad()
    def take_transition(self, state, top_k: int = 30):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        outputs = self.nn.encoder(states)
        q_z, q_m, q_s, t = outputs
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        z_latent = q_z.cpu().numpy()
        z_future = z_latent + 1e-2 * grads
        distances = pairwise_distances(z_latent, z_future)
        sigma = np.median(distances)
        similarity = np.exp(-(distances**2) / (2 * sigma**2))
        transition_matrix = similarity / similarity.sum(axis=1, keepdims=True)

        def sparsify_transitions(trans_matrix, top_k=top_k):
            n_cells = trans_matrix.shape[0]
            sparse_trans = np.zeros_like(trans_matrix)
            for i in range(n_cells):
                top_indices = np.argsort(trans_matrix[i])[::-1][:top_k]
                sparse_trans[i, top_indices] = trans_matrix[i, top_indices]
                sparse_trans[i] /= sparse_trans[i].sum()
            return sparse_trans

        transition_matrix = sparsify_transitions(transition_matrix)
        return transition_matrix

    def update(self, *states):
        # Initialize MoCo loss
        moco_loss = torch.zeros(1).to(self.device)

        if self.use_moco:
            states_x = torch.tensor(states[1], dtype=torch.float).to(self.device)
            states_q = torch.tensor(states[1], dtype=torch.float).to(self.device)
            states_k = torch.tensor(states[2], dtype=torch.float).to(self.device)
            outputs = self.nn(states_x, states_q, states_k)
        else:
            states_q = torch.tensor(states[0], dtype=torch.float).to(self.device)
            outputs = self.nn(states_q)

        if self.use_ode:
            if self.loss_mode == "zinb":
                if self.use_moco:
                    (
                        q_z,
                        q_m,
                        q_s,
                        x,
                        pred_x,
                        dropout_logits,
                        le,
                        le_ode,
                        pred_xl,
                        dropout_logitsl,
                        q_z_ode,
                        pred_x_ode,
                        dropout_logits_ode,
                        pred_xl_ode,
                        dropout_logitsl_ode,
                        logits,
                        labels
                    ) = outputs
                    moco_loss = self.moco_criterion(logits, labels)
                else:
                    (
                        q_z,
                        q_m,
                        q_s,
                        x,
                        pred_x,
                        dropout_logits,
                        le,
                        le_ode,
                        pred_xl,
                        dropout_logitsl,
                        q_z_ode,
                        pred_x_ode,
                        dropout_logits_ode,
                        pred_xl_ode,
                        dropout_logitsl_ode,
                    ) = outputs
                    
                qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()

                l = x.sum(-1).view(-1, 1)
                pred_x = pred_x * l
                pred_x_ode = pred_x_ode * l
                disp = torch.exp(self.nn.decoder.disp)
                recon_loss = (
                    -self._log_zinb(x, pred_x, disp, dropout_logits).sum(-1).mean()
                )
                recon_loss += (
                    -self._log_zinb(x, pred_x_ode, disp, dropout_logits_ode)
                    .sum(-1)
                    .mean()
                )

                if self.irecon:
                    pred_xl = pred_xl * l + 1e-8
                    pred_xl_ode = pred_xl_ode * l
                    irecon_loss = (
                        -self.irecon
                        * self._log_zinb(x, pred_xl, disp, dropout_logitsl)
                        .sum(-1)
                        .mean()
                    )
                    irecon_loss += (
                        -self.irecon
                        * self._log_zinb(x, pred_xl_ode, disp, dropout_logitsl_ode)
                        .sum(-1)
                        .mean()
                    )
                else:
                    irecon_loss = torch.zeros(1).to(self.device)

            else:
                if self.use_moco:
                    (
                        q_z,
                        q_m,
                        q_s,
                        x,
                        pred_x,
                        le,
                        le_ode,
                        pred_xl,
                        q_z_ode,
                        pred_x_ode,
                        pred_xl_ode,
                        logits,
                        labels
                    ) = outputs
                    moco_loss = self.moco_criterion(logits, labels)
                else:
                    (
                        q_z,
                        q_m,
                        q_s,
                        x,
                        pred_x,
                        le,
                        le_ode,
                        pred_xl,
                        q_z_ode,
                        pred_x_ode,
                        pred_xl_ode,
                    ) = outputs
                    
                qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()

                if self.loss_mode == "nb":
                    l = x.sum(-1).view(-1, 1)
                    pred_x = pred_x * l
                    pred_x_ode = pred_x_ode * l
                    disp = torch.exp(self.nn.decoder.disp)
                    recon_loss = -self._log_nb(x, pred_x, disp).sum(-1).mean()
                    recon_loss += -self._log_nb(x, pred_x_ode, disp).sum(-1).mean()

                    if self.irecon:
                        pred_xl = pred_xl * l
                        pred_xl_ode = pred_xl_ode * l
                        irecon_loss = (
                            -self.irecon * self._log_nb(x, pred_xl, disp).sum(-1).mean()
                        )
                        irecon_loss += (
                            -self.irecon
                            * self._log_nb(x, pred_xl_ode, disp).sum(-1).mean()
                        )
                    else:
                        irecon_loss = torch.zeros(1).to(self.device)

                else:
                    recon_loss = F.mse_loss(x, pred_x, reduction="none").sum(-1).mean()
                    recon_loss += (
                        F.mse_loss(x, pred_x_ode, reduction="none").sum(-1).mean()
                    )
                    if self.irecon:
                        irecon_loss = (
                            self.irecon * F.mse_loss(x, pred_xl, reduction="none").sum(-1).mean()
                        )
                        irecon_loss += (
                            self.irecon * F.mse_loss(x, pred_xl_ode, reduction="none").sum(-1).mean()
                        )
                    else:
                        irecon_loss = torch.zeros(1).to(self.device)

            p_m = torch.zeros_like(q_m)
            p_s = torch.zeros_like(q_s)

            kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()

            if self.dip:
                dip_loss = self.dip * self._dip_loss(q_m, q_s)
            else:
                dip_loss = torch.zeros(1).to(self.device)

            if self.tc:
                tc_loss = self.tc * self._betatc_compute_total_correlation(
                    q_z, q_m, q_s
                )
            else:
                tc_loss = torch.zeros(1).to(self.device)

            if self.info:
                mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z))
            else:
                mmd_loss = torch.zeros(1).to(self.device)

            total_loss = (
                self.recon * recon_loss
                + irecon_loss
                + qz_div
                + kl_div
                + dip_loss
                + tc_loss
                + mmd_loss
                + self.moco_weight * moco_loss
            )

        else:
            if self.loss_mode == "zinb":
                if self.use_moco:
                    q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl, logits, labels = outputs
                    moco_loss = self.moco_criterion(logits, labels)
                else:
                    q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl = outputs

                l = states_q.sum(-1).view(-1, 1)
                pred_x = pred_x * l + 1e-8

                disp = torch.exp(self.nn.decoder.disp)
                recon_loss = (
                    -self._log_zinb(states_q, pred_x, disp, dropout_logits).sum(-1).mean()
                )

                if self.irecon:
                    pred_xl = pred_xl * l + 1e-8
                    irecon_loss = (
                        -self.irecon
                        * self._log_zinb(states_q, pred_xl, disp, dropout_logitsl)
                        .sum(-1)
                        .mean()
                    )
                else:
                    irecon_loss = torch.zeros(1).to(self.device)

            else:
                if self.use_moco:
                    q_z, q_m, q_s, pred_x, le, pred_xl, logits, labels = outputs
                    moco_loss = self.moco_criterion(logits, labels)
                else:
                    q_z, q_m, q_s, pred_x, le, pred_xl = outputs

                if self.loss_mode == "nb":
                    l = states_q.sum(-1).view(-1, 1)
                    pred_x = pred_x * l + 1e-8

                    disp = torch.exp(self.nn.decoder.disp)
                    recon_loss = -self._log_nb(states_q, pred_x, disp).sum(-1).mean()

                    if self.irecon:
                        pred_xl = pred_xl * l
                        irecon_loss = (
                            -self.irecon
                            * self._log_nb(states_q, pred_xl, disp).sum(-1).mean()
                        )
                    else:
                        irecon_loss = torch.zeros(1).to(self.device)

                else:
                    recon_loss = (
                        F.mse_loss(states_q, pred_x, reduction="none").sum(-1).mean()
                    )
                    if self.irecon:
                        irecon_loss = (
                            self.irecon * F.mse_loss(states_q, pred_xl, reduction="none").sum(-1).mean()
                        )
                    else:
                        irecon_loss = torch.zeros(1).to(self.device)

            p_m = torch.zeros_like(q_m)
            p_s = torch.zeros_like(q_s)

            kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()

            if self.dip:
                dip_loss = self.dip * self._dip_loss(q_m, q_s)
            else:
                dip_loss = torch.zeros(1).to(self.device)

            if self.tc:
                tc_loss = self.tc * self._betatc_compute_total_correlation(
                    q_z, q_m, q_s
                )
            else:
                tc_loss = torch.zeros(1).to(self.device)

            if self.info:
                mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z))
            else:
                mmd_loss = torch.zeros(1).to(self.device)

            total_loss = (
                self.recon * recon_loss
                + irecon_loss
                + kl_div
                + dip_loss
                + tc_loss
                + mmd_loss
                + self.moco_weight * moco_loss
            )

        self.nn_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 1.0)
        self.nn_optimizer.step()

        self.loss.append(
            (
                total_loss.item(),
                recon_loss.item(),
                irecon_loss.item(),
                kl_div.item(),
                dip_loss.item(),
                tc_loss.item(),
                mmd_loss.item(),
                moco_loss.item(),  
            )
        )

