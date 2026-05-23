import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class TripletSILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[],
            accelerator=None,
            latents_scale=None,
            latents_bias=None,
            contrastive_weight=0.05,
            null_class_idx=1000,
            dont_contrast_on_unconditional=False,
            is_class_conditioned=False,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.contrastive_weight = contrastive_weight
        self.null_class_idx = null_class_idx
        self.dont_contrast_on_unconditional = dont_contrast_on_unconditional
        self.is_class_conditioned = is_class_conditioned

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t = 1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs is None:
            model_kwargs = {}
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        else:
            rnd_normal = torch.randn((images.shape[0], 1, 1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            else:
                time_input = 2 / np.pi * torch.atan(sigma)
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        model_input = alpha_t * images + sigma_t * noises
        model_target = d_alpha_t * images + d_sigma_t * noises
        
        # 模型返回 (model_output, zs_tilde, labels)
        model_output, zs_tilde, labels = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # 计算对比损失（核心修改）
        contrastive_loss = torch.tensor(0., device=images.device)
        flow_loss = denoising_loss

        if self.is_class_conditioned and self.contrastive_weight > 0:
            labels = labels.long()
            valid_mask = labels != self.null_class_idx
            if self.dont_contrast_on_unconditional:
                # 仅在条件样本上执行对比
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            else:
                valid_indices = torch.arange(images.size(0), device=images.device)

            if len(valid_indices) > 1:
                if len(zs) > 0:
                    # 有编码器：使用 zs 特征
                    z = zs[0][valid_indices]
                    z_tilde_local = zs_tilde[0][valid_indices]
                    z_norm = F.normalize(z, dim=-1)
                    z_tilde_norm = F.normalize(z_tilde_local, dim=-1)
                    contrastive_loss = mean_flat(-(z_norm * z_tilde_norm).sum(dim=-1))
                else:
                    # 无编码器：直接对比 flow 向量（论文核心思想）
                    flow_pred = model_output[valid_indices].reshape(len(valid_indices), -1)
                    flow_target = model_target[valid_indices].reshape(len(valid_indices), -1)
                    flow_pred_norm = F.normalize(flow_pred, dim=-1)
                    flow_target_norm = F.normalize(flow_target, dim=-1)
                    contrastive_loss = mean_flat(-(flow_pred_norm * flow_target_norm).sum(dim=-1))
            else:
                contrastive_loss = torch.tensor(0., device=images.device)

        total_loss = denoising_loss + self.contrastive_weight * contrastive_loss
        return {'loss': total_loss, 'flow_loss': flow_loss, 'contrastive_loss': contrastive_loss}, torch.tensor(0., device=images.device)
