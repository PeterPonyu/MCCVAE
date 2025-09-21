import torch
import torch.nn.functional as F
from torchdiffeq import odeint
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from typing import Optional


class scviMixin:
    def _normal_kl(self, mu1, lv1, mu2, lv2):
        """
        计算两个正态分布之间的KL散度

        参数:
        mu1, mu2: 两个分布的均值
        lv1, lv2: 两个分布的对数方差

        返回:
        KL散度值
        """
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.0
        lstd2 = lv2 / 2.0
        kl = lstd2 - lstd1 + (v1 + (mu1 - mu2) ** 2.0) / (2.0 * v2) - 0.5
        return kl

    def _log_nb(self, x, mu, theta, eps=1e-8):
        """
        计算负二项分布下的对数概率

        参数:
        x: 数据
        mu: 分布均值
        theta: 离散参数
        eps: 数值稳定性常数

        返回:
        对数概率值
        """
        log_theta_mu_eps = torch.log(theta + mu + eps)
        res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        return res

    def _log_zinb(self, x, mu, theta, pi, eps=1e-8):
        """
        计算零膨胀负二项分布下的对数概率

        参数:
        x: 数据
        mu: 分布均值
        theta: 离散参数
        pi: 零膨胀混合权重的logits
        eps: 数值稳定性常数

        返回:
        对数概率值
        """
        softplus_pi = F.softplus(-pi)
        log_theta_eps = torch.log(theta + eps)
        log_theta_mu_eps = torch.log(theta + mu + eps)
        pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

        case_zero = F.softplus(pi_theta_log) - softplus_pi
        mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

        case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
        mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

        res = mul_case_zero + mul_case_non_zero
        return res


class NODEMixin:
    """
    Mixin类，提供Neural ODE相关的功能
    """

    @staticmethod
    def get_step_size(step_size, t0, t1, n_points):
        """
        获取ODE求解器的步长配置

        参数:
        step_size: 步长，如果为None则自动计算
        t0: 起始时间
        t1: 结束时间
        n_points: 时间点数量

        返回:
        ODE求解器的配置字典
        """
        if step_size is None:
            return {}
        else:
            if step_size == "auto":
                step_size = (t1 - t0) / (n_points - 1)
            return {"step_size": step_size}

    def solve_ode(
        self,
        ode_func: torch.nn.Module,
        z0: torch.Tensor,
        t: torch.Tensor,
        method: str = "rk4",
        step_size: Optional[float] = None,
    ) -> torch.Tensor:
        """
        使用torchdiffeq求解ODE

        参数:
        ode_func: ODE函数模型
        z0: 初始状态
        t: 时间点
        method: 求解方法
        step_size: 步长

        返回:
        ODE求解结果
        """
        options = self.get_step_size(step_size, t[0], t[-1], len(t))

        # 确保数据在CPU上，因为某些ODE求解器在GPU上可能会有问题
        cpu_z0 = z0.to("cpu")
        cpu_t = t.to("cpu")

        # 求解ODE
        pred_z = odeint(ode_func, cpu_z0, cpu_t, method=method, options=options)

        # 将结果移回到原始设备
        pred_z = pred_z.to(z0.device)

        return pred_z


class betatcMixin:
    def _betatc_compute_gaussian_log_density(self, samples, mean, log_var):
        import math

        pi = torch.tensor(math.pi, requires_grad=False)
        normalization = torch.log(2 * pi)
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)

    def _betatc_compute_total_correlation(self, z_sampled, z_mean, z_logvar):
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            z_sampled.unsqueeze(dim=1),
            z_mean.unsqueeze(dim=0),
            z_logvar.unsqueeze(dim=0),
        )
        log_qz_product = log_qz_prob.exp().sum(dim=1).log().sum(dim=1)
        log_qz = log_qz_prob.sum(dim=2).exp().sum(dim=1).log()
        return (log_qz - log_qz_product).mean()


class infoMixin:
    def _compute_mmd(self, z_posterior_samples, z_prior_samples):
        mean_pz_pz = self._compute_unbiased_mean(
            self._compute_kernel(z_prior_samples, z_prior_samples), unbaised=True
        )
        mean_pz_qz = self._compute_unbiased_mean(
            self._compute_kernel(z_prior_samples, z_posterior_samples), unbaised=False
        )
        mean_qz_qz = self._compute_unbiased_mean(
            self._compute_kernel(z_posterior_samples, z_posterior_samples),
            unbaised=True,
        )
        mmd = mean_pz_pz - 2 * mean_pz_qz + mean_qz_qz
        return mmd

    def _compute_unbiased_mean(self, kernel, unbaised):
        N, M = kernel.shape
        if unbaised:
            sum_kernel = kernel.sum(dim=(0, 1)) - torch.diagonal(
                kernel, dim1=0, dim2=1
            ).sum(dim=-1)
            mean_kernel = sum_kernel / (N * (N - 1))
        else:
            mean_kernel = kernel.mean(dim=(0, 1))
        return mean_kernel

    def _compute_kernel(self, z0, z1):
        batch_size, z_size = z0.shape
        z0 = z0.unsqueeze(-2)
        z1 = z1.unsqueeze(-3)
        z0 = z0.expand(batch_size, batch_size, z_size)
        z1 = z1.expand(batch_size, batch_size, z_size)
        kernel = self._kernel_rbf(z0, z1)
        return kernel

    def _kernel_rbf(self, x, y):
        z_size = x.shape[-1]
        sigma = 2 * 2 * z_size
        kernel = torch.exp(-((x - y).pow(2).sum(dim=-1) / sigma))
        return kernel


class dipMixin:
    def _dip_loss(self, q_m, q_s):
        cov_matrix = self._dip_cov_matrix(q_m, q_s)
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        dip_loss_d = torch.sum((cov_diag - 1) ** 2)
        dip_loss_od = torch.sum(cov_off_diag**2)
        dip_loss = 10 * dip_loss_d + 5 * dip_loss_od
        return dip_loss

    def _dip_cov_matrix(self, q_m, q_s):
        cov_q_mean = torch.cov(q_m.T)
        E_var = torch.mean(torch.diag(F.softplus(q_s).exp()), dim=0)
        cov_matrix = cov_q_mean + E_var
        return cov_matrix


class envMixin:
    def _calc_score(self, latent, ifall=False):
        n = latent.shape[1]
        labels = self._calc_label(latent)
        scores = self._metrics(latent, labels, ifall)
        return scores

    def _calc_label(self, latent):
        labels = KMeans(latent.shape[1]).fit_predict(latent)
        return labels

    def _calc_corr(self, latent):
        acorr = abs(np.corrcoef(latent.T))
        return acorr.sum(axis=1).mean().item() - 1

    def _metrics(self, latent, labels, ifall):
        ARI = adjusted_mutual_info_score(self.labels if ifall else self.labels[self.idx], labels)
        NMI = normalized_mutual_info_score(self.labels if ifall else self.labels[self.idx], labels)
        ASW = silhouette_score(latent, labels)
        C_H = calinski_harabasz_score(latent, labels)
        D_B = davies_bouldin_score(latent, labels)
        P_C = self._calc_corr(latent)
        return ARI, NMI, ASW, C_H, D_B, P_C
