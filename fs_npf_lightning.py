import sys
from typing import Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning.core.module import LightningModule
from torch.utils.data import DataLoader

sys.path.append("./models/")

import matplotlib as mpl
import numpy as np
from scipy.stats import iqr

from models.flow_npf_model import FlowNumPFNet
from models.sampler import pndm_sampler
from utils.datasetloader import FastSimDataset, VarTransform
from utils.lion_opt import Lion

mpl.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
import copy

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from utils.conditional_flow_matching import TargetConditionalFlowMatcher

from utils.custom_scheduler import CosineAnnealingWarmupRestarts

def normalize_phi(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


def subplots_with_absolute_sized_axes(
    nrows: int,
    ncols: int,
    figsize: Tuple[float, float],
    axis_width: float,
    axis_height: float,
    sharex: bool = False,
    sharey: bool = False,
    **kwargs,
) -> Tuple[plt.Figure, np.ndarray]:
    """Create axes with exact sizes.

    Spaces axes as far from each other and the figure edges as possible
    within the grid defined by nrows, ncols, and figsize.

    Allows you to share y and x axes, if desired.
    """
    fig = plt.figure(figsize=figsize, **kwargs)
    figwidth, figheight = figsize
    # spacing on each left and right side of the figure
    h_margin = (figwidth - (ncols * axis_width)) / figwidth / ncols / 2
    # spacing on each top and bottom of the figure
    v_margin = (figheight - (nrows * axis_height)) / figheight / nrows / 2
    row_addend = 1 / nrows
    col_addend = 1 / ncols
    inner_ax_width = axis_width / figwidth
    inner_ax_height = axis_height / figheight
    axes = []
    sharex_ax = None
    sharey_ax = None
    for row in range(nrows):
        bottom = (row * row_addend) + v_margin
        for col in range(ncols):
            left = (col * col_addend) + h_margin
            if not axes:
                axes.append(
                    fig.add_axes([left, bottom, inner_ax_width, inner_ax_height])
                )
                if sharex:
                    sharex_ax = axes[0]
                if sharey:
                    sharey_ax = axes[0]
            else:
                axes.append(
                    fig.add_axes(
                        [left, bottom, inner_ax_width, inner_ax_height],
                        sharex=sharex_ax,
                        sharey=sharey_ax,
                    )
                )
    return fig, np.flip(np.asarray(list(axes)).reshape((nrows, ncols)), axis=0)

class FlowNumPFLightning(LightningModule):
    def __init__(self, config, comet_logger=None):
        super().__init__()
        torch.manual_seed(1)
        self.config = config

        self.loss_type = config.get("loss_type", "mse")
        self.loss = torch.nn.MSELoss()

        self.sigma = config["sigma"]
        self.flow_type = config["flow_type"]
        self.n_steps = config["n_steps"]
        self.time_pow = config.get("time_pow", False)
        self.opt = config.get("opt", "adamw")
        self.noisy_dim = config.get("noisy_dim", 4)
        self.ext_class = config.get("ext_class", False)

        self.net = FlowNumPFNet(config, noisy_dim=self.noisy_dim)

        # if self.flow_type == "otcfm":
        #     self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=self.sigma)
        self.FM = TargetConditionalFlowMatcher(sigma=self.sigma)

        self.comet_logger = comet_logger

        self.validation_step_outputs = []
        self.var_transform_dict = {
            key: VarTransform(key, val)
            for key, val in self.config["var_transform"].items()
        }

    def set_comet_logger(self, comet_logger):
        self.comet_logger = comet_logger

    def forward(self, data):
        truth, pflow, mask, global_data = data
        x0 = torch.randn_like(pflow)
        if self.time_pow:
            t = torch.Tensor(np.random.power(3, size=pflow.shape[0])).to(
                pflow.device
            )
        else:
            t = None
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, pflow, t=t)

        vt = self.net(xt, truth, mask, timestep=t, global_data=global_data)

        ht_loss = self.loss(ut[..., 0], vt[..., 0])
        npf_loss = self.loss(ut[..., 1], vt[..., 1])
        met_x_loss = self.loss(ut[..., 2], vt[..., 2])
        met_y_loss = self.loss(ut[..., 3], vt[..., 3])

        losses = {
            "ht_loss": ht_loss,
            "npf_loss": npf_loss,
            "met_x_loss": met_x_loss,
            "met_y_loss": met_y_loss,
        }
        total_loss = ht_loss + npf_loss + met_x_loss + met_y_loss
        total_loss /= 4
        return total_loss, losses

    @torch.no_grad()
    def sample(
        self,
        truth,
        pflow_shape,
        mask,
        n_steps=None,
        method="pdnm",
        global_data=None,
        dt=0.0,
        save_seq=False,
    ):
        if n_steps is None:
            n_steps = self.n_steps
        return pndm_sampler(
            self.net,
            truth,
            pflow_shape,
            mask,
            global_data,
            n_steps=n_steps,
            dt=dt,
            save_seq=save_seq,
            zero_init_padded=False,
            reverse_time=self.flow_type == "rcfm",
        )[0]

    def training_step(self, data, batch_idx):
        total_loss, losses = self.forward(data)
        losses["total_loss"] = total_loss

        self.log_dict(
            {f"train/{k}": v.item() for k, v in losses.items()},
            batch_size=data[0].shape[0],
            sync_dist=True,
        )

        return total_loss

    def validation_step(self, data, batch_idx):
        total_loss, losses = self.forward(data)
        losses["total_loss"] = total_loss

        self.log_dict(
            {f"val/{k}": v.item() for k, v in losses.items()},
            batch_size=data[0].shape[0],
            sync_dist=True,
        )
        truth, pflow, mask, global_data = data
        return_dict = {}
        with torch.no_grad():
            pred = self.sample(
                truth,
                pflow.shape,
                mask,
                global_data=global_data,
                method="pndm",
            )
        pf_ht = self.var_transform_dict["ht"].inverse_transform(pflow[..., 0]).cpu()
        fs_ht = self.var_transform_dict["ht"].inverse_transform(pred[..., 0])
        tr_ht = (
            self.var_transform_dict["ht"]
            .inverse_transform(global_data[..., -1 - 5 * self.ext_class])
            .cpu()
        )

        n_pf = (
            self.var_transform_dict["npart"]
            .inverse_transform(pflow[..., 1].cpu())
            .round()
        )
        n_fs = (
            self.var_transform_dict["npart"]
            .inverse_transform(pred[..., 1].cpu())
            .round()
        )
        n_tr = mask.sum(-1).float().cpu()

        pf_met_x = (
            self.var_transform_dict["met_x"].inverse_transform(pflow[..., 2]).cpu()
        )
        fs_met_x = self.var_transform_dict["met_x"].inverse_transform(pred[..., 2])
        tr_met_x = (
            self.var_transform_dict["met_x"]
            .inverse_transform(global_data[..., -4 - 5 * self.ext_class])
            .cpu()
        )

        pf_met_y = (
            self.var_transform_dict["met_y"].inverse_transform(pflow[..., 3]).cpu()
        )
        fs_met_y = self.var_transform_dict["met_y"].inverse_transform(pred[..., 3])
        tr_met_y = (
            self.var_transform_dict["met_y"]
            .inverse_transform(global_data[..., -3 - 5 * self.ext_class])
            .cpu()
        )

        pred_losses = {
            "ht": self.loss(pf_ht, fs_ht),
            "npf": self.loss(n_pf, n_fs),
            "met_x": self.loss(pf_met_x, fs_met_x),
            "met_y": self.loss(pf_met_y, fs_met_y),
        }
        if self.ext_class:
            for i in range(5):
                pred_losses[f"class_{i}"] = self.loss(
                    self.var_transform_dict[f"class_{i}"]
                    .inverse_transform(pflow[..., 4 + i])
                    .cpu(),
                    self.var_transform_dict[f"class_{i}"]
                    .inverse_transform(pred[..., 4 + i])
                    .cpu(),
                )

        self.log_dict(
            {f"val/pred_{k}": v.item() for k, v in pred_losses.items()},
            batch_size=data[0].shape[0],
            sync_dist=True,
        )

        return_dict["pf_ht"] = pf_ht.cpu()
        return_dict["fs_ht"] = fs_ht.cpu()
        return_dict["tr_ht"] = tr_ht.cpu()

        return_dict["pf_met_x"] = pf_met_x.cpu()
        return_dict["fs_met_x"] = fs_met_x.cpu()
        return_dict["tr_met_x"] = tr_met_x.cpu()

        return_dict["pf_met_y"] = pf_met_y.cpu()
        return_dict["fs_met_y"] = fs_met_y.cpu()
        return_dict["tr_met_y"] = tr_met_y.cpu()

        return_dict["n_pf"] = n_pf.cpu()
        return_dict["n_fs"] = n_fs.cpu()
        return_dict["n_tr"] = n_tr.cpu()

        return_dict["total_loss"] = total_loss.item()

        self.validation_step_outputs.append(return_dict)

    def on_train_epoch_end(self):
        if self.config["lr_scheduler"] is not False:
            self.log("lr", self.lr_schedulers().get_last_lr()[0], sync_dist=True)
            self.lr_schedulers().step()

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        losses = np.mean([x["total_loss"] for x in outputs])

        self.log("val_loss_avg", losses, sync_dist=True)

        if self.config["lr_scheduler"] is not False:
            self.log("lr", self.lr_schedulers().get_last_lr()[0], sync_dist=True)

        pf_ht = torch.cat([x["pf_ht"] for x in outputs], dim=0)
        fs_ht = torch.cat([x["fs_ht"] for x in outputs], dim=0)
        tr_ht = torch.cat([x["tr_ht"] for x in outputs], dim=0)

        pf_met_x = torch.cat([x["pf_met_x"] for x in outputs], dim=0)
        fs_met_x = torch.cat([x["fs_met_x"] for x in outputs], dim=0)
        tr_met_x = torch.cat([x["tr_met_x"] for x in outputs], dim=0)

        pf_met_y = torch.cat([x["pf_met_y"] for x in outputs], dim=0)
        fs_met_y = torch.cat([x["fs_met_y"] for x in outputs], dim=0)
        tr_met_y = torch.cat([x["tr_met_y"] for x in outputs], dim=0)

        n_pf = torch.cat([x["n_pf"] for x in outputs], dim=0)
        n_fs = torch.cat([x["n_fs"] for x in outputs], dim=0)
        n_tr = torch.cat([x["n_tr"] for x in outputs], dim=0)
        n_fs[n_fs < 1] = n_tr[n_fs < 1]

        pfs = torch.stack([pf_ht, pf_met_x, pf_met_y, n_pf], dim=-1).numpy()
        fss = torch.stack([fs_ht, fs_met_x, fs_met_y, n_fs], dim=-1).numpy()
        trs = torch.stack([tr_ht, tr_met_x, tr_met_y, n_tr], dim=-1).numpy()

        self.log_image(pfs, fss, trs)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        if self.opt == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=float(self.config["learningrate"])
            )
        elif self.opt == "lion":
            optimizer = Lion(
                self.parameters(),
                lr=float(self.config["learningrate"]),
            )
        if self.config["lr_scheduler"] is False:
            return optimizer
        else:
            print("\nscheduler exists!\n")
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=10,
                warmup_steps=4,
                max_lr=4 * float(self.config["learningrate"]),
                min_lr=1e-5,
                gamma=0.8,
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        if "reduce_ds_train" in self.config:
            reduce_ds = self.config["reduce_ds_train"]
        else:
            reduce_ds = 1

        dataset = FastSimDataset(
            self.config["truth_path_train"],
            self.config,
            reduce_ds=reduce_ds,
            entry_start=self.config["entry_start_train"],
        )
        loader = DataLoader(
            dataset,
            num_workers=self.config["num_workers"],
            batch_size=self.config["batchsize"],
            drop_last=True,
            shuffle=True,
            pin_memory=False,
            prefetch_factor=4,
        )
        return loader

    def val_dataloader(self):
        if "reduce_ds_valid" in self.config:
            reduce_ds = self.config["reduce_ds_valid"]
        else:
            reduce_ds = 1
        dataset = FastSimDataset(
            self.config["truth_path_valid"],
            self.config,
            reduce_ds=reduce_ds,
            entry_start=self.config["entry_start_valid"],
        )

        loader = DataLoader(
            dataset,
            num_workers=0,
            batch_size=self.config.get("val_batchsize", self.config["batchsize"]),
            drop_last=True,
            pin_memory=False,
            shuffle=False,
        )
        return loader

    def log_image(
        self,
        pf,
        fs,
        tr,
    ):
        # fig, axs = plt.subplots(2, 3, figsize=(16, 8), dpi=200)#, tight_layout=True)
        fig, axs = subplots_with_absolute_sized_axes(
            4, 2, figsize=(18, 18), axis_width=8, axis_height=4, dpi=200
        )
        canvas = FigureCanvas(fig)
        cmap = copy.copy(plt.get_cmap("PuRd"))
        cmap.set_under("white")

        def _add_to_hist(ax, data, bins, label, **kwargs):
            ax.hist(
                data,
                bins=bins,
                label=f"{label} {np.mean(data):.2f}/{np.std(data):.2f}/{iqr(data):.2f}",
                **kwargs,
            )

        bins = {
            "ht": np.linspace(0, 3000, 100),
            "met_x": np.linspace(-100, 100, 100),
            "met_y": np.linspace(-100, 100, 100),
            "npart": np.linspace(0.5, 400.5, 100),
        }
        res_bins = {
            "ht": np.linspace(-2, 1, 50),
            "npart": np.linspace(-250, 150, 50),
            "met_x": np.linspace(-100, 100, 50),
            "met_y": np.linspace(-100, 100, 50),
        }
        labels = ["HT", "MET_x", "MET_y", "Npart"]
        for i, var in enumerate(["ht", "met_x", "met_y", "npart"]):
            _add_to_hist(
                axs[i][0], pf[:, i], bins=bins[var], label=f"PF", histtype="stepfilled"
            )
            _add_to_hist(
                axs[i][0], fs[:, i], bins=bins[var], label=f"FS", histtype="step"
            )
            _add_to_hist(
                axs[i][0],
                tr[:, i],
                bins=bins[var],
                label=f"TR",
                histtype="step",
                color="k",
            )
            axs[i][0].set_title(f"{labels[i]}")
            axs[i][0].legend()

            pf_res = (pf[:, i] - tr[:, i]) / pf[:, i] if i == 0 else pf[:, i] - tr[:, i]
            fs_res = (fs[:, i] - tr[:, i]) / fs[:, i] if i == 0 else fs[:, i] - tr[:, i]
            _add_to_hist(
                axs[i][1],
                pf_res,
                bins=res_bins[var],
                label=f"PF-TR",
                histtype="stepfilled",
            )
            _add_to_hist(
                axs[i][1], fs_res, bins=res_bins[var], label=f"FS-TR", histtype="step"
            )
            axs[i][1].set_title(f"{labels[i]} Residuals")
            axs[i][1].legend()

        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(h), int(w), 3
        )
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        plt.savefig(f"ED_{self.current_epoch}.png")
        if self.logger is not None:
            self.logger.experiment.log_image(
                image_data=image,
                name=f"ED_{self.current_epoch}",
                overwrite=False,
                image_format="png",
            )
        else:
            plt.savefig(f"ED_{self.current_epoch}.png")
        plt.close(fig)