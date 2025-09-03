import torch
from ase import Atoms
import wandb

class DensityLoss:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.use_wandb = config['wandb']
        self.density_loss_fn = torch.nn.L1Loss(reduction='sum')
        self.r2loss = torch.nn.L1Loss()
        self.loss_type = config['loss_type']
        self.loss_coef = torch.tensor(config['loss_coef'], device=device)
        self.huber_loss_threshold = torch.tensor(config['huber_loss_threshold'], device=device)

    def compute_integrated_error(self, pred_cd, true_cd_tens, dv, n_valence_electrons, sampled_points=None):
        if sampled_points is None:
            delta = self.density_loss_fn(pred_cd, true_cd_tens)
        else:
            delta = torch.abs(pred_cd.flatten()[sampled_points] - true_cd_tens.flatten()[sampled_points]).sum()
        # percent_error = torch.abs(pred_cd-true_cd_tens).sum()*dv/n_valence_electrons
        #integrated_error = (delta * dv) / n_valence_electrons
        epsilon = torch.finfo(true_cd_tens.dtype).eps
        integrated_error = delta / (true_cd_tens.sum() + epsilon)
        if torch.isnan(integrated_error) or torch.isinf(integrated_error):
            integrated_error = torch.zeros_like(integrated_error)
        else:
            integrated_error = torch.clamp(integrated_error, min=0.0)
        return integrated_error

    def compute_R2_loss(self, pred_cd: torch.tensor, true_cd_tens: torch.tensor):
        pred_cd = pred_cd.to(self.device)
        true_cd_tens = true_cd_tens.to(self.device)
        R2_loss = self.r2loss(pred_cd.flatten(), true_cd_tens.flatten())
        return R2_loss


    def compute_total_loss(self, sys: Atoms,
                           pred_cd: torch.tensor,
                           true_cd_tens: torch.tensor,
                           grid_dict: dict,
                           n_valence_electrons: int,
                           sampled_points: torch.tensor = None,
                           training: bool = True,
                           mol_volume: float = None):
        pred_cd = pred_cd.to(self.device)
        true_cd_tens = true_cd_tens.to(self.device)
        if sampled_points is not None:
            sampled_points = sampled_points.to(self.device)
        grid_points = grid_dict["nx"] * grid_dict["ny"] * grid_dict["nz"]
        if mol_volume is not None:
            dv = mol_volume / grid_points
        else:
            dv = sys.get_volume() / grid_points

        integrated_error = self.compute_integrated_error(pred_cd, true_cd_tens, dv, n_valence_electrons, sampled_points)
        integrated_error = integrated_error.to(self.device)
        self.huber_loss_threshold = self.huber_loss_threshold.to(self.device)
        if self.loss_type == "huber":
            loss = torch.where(
                integrated_error < self.huber_loss_threshold,
                self.loss_coef * 0.5 * integrated_error ** 2,
                self.loss_coef * self.huber_loss_threshold * (integrated_error - 0.5 * self.huber_loss_threshold)
            )
        else:
            #loss = self.compute_R2_loss(pred_cd, true_cd_tens)
            loss = integrated_error

        if self.use_wandb and training:
            wandb.log({
                "Density Loss": loss.item(),
            })

        return loss, integrated_error

