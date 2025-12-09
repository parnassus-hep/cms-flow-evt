import sys
from typing import Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning.core.module import LightningModule
from torch.utils.data import DataLoader

sys.path.append("./models/")

import matplotlib as mpl
import numpy as np

from models.flow_model import FlowNet
from set2setloss import Set2SetLoss
from utils.datasetloader import FastSimDataset, VarTransform
from utils.lion_opt import Lion

mpl.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
import copy

import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix

from models.dpm import DPM_Solver, NoiseScheduleFlow
from utils.conditional_flow_matching import *
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


class MSEAndDirectionLoss(torch.nn.Module):
    """
    Figure 7 - https://arxiv.org/abs/2410.10356
    """

    def __init__(self, cosine_sim_dim: int = 2):
        super().__init__()
        assert cosine_sim_dim > 0, "cannot be batch dimension"
        self.cosine_sim_dim = cosine_sim_dim

    def forward(self, pred, target, **kwargs):
        mse_loss = F.mse_loss(pred, target, reduction="sum")
        direction_loss = (
            1.0 - F.cosine_similarity(pred, target, dim=self.cosine_sim_dim)
        ).sum()
        return mse_loss + direction_loss


class FlowLightning(LightningModule):
    def __init__(self, config, comet_logger=None):
        super().__init__()
        torch.manual_seed(1)
        self.config = config

        self.loss_type = config.get("loss_type", "mse")
        self.loss = MSEAndDirectionLoss()

        self.pred_loss = Set2SetLoss()

        self.n_steps = config["n_steps"]

        self.opt = config.get("opt", "adamw")

        self.net = FlowNet(config)
        self.FM = ReverseConditionalFlowMatcher(sigma=0.0)

        def model_fn(x, timestep, truth, mask, global_data):
            return (1 - timestep.view(-1, 1, 1)) * self.net(
                x, truth, mask, timestep, global_data
            ) + x

        self.dpm = DPM_Solver(model_fn=model_fn, noise_schedule=NoiseScheduleFlow())

        self.pflow_variables = [
            el
            for el in self.config.get(
                "pflow_variables",
                [
                    "pflow_pt",
                    "pflow_eta",
                    "pflow_phi",
                ],
            )
        ]
        for i, var in enumerate(self.pflow_variables):
            if var == "pflow_pt":
                self.pflow_variables[i] = "pflow_ptrel"

        self.var_transform_dict = {
            key: VarTransform(key, val)
            for key, val in self.config["var_transform"].items()
        }

        self.comet_logger = comet_logger

        self.validation_step_outputs = []

    def set_comet_logger(self, comet_logger):
        self.comet_logger = comet_logger

    def forward(self, data):
        truth, pflow, mask, global_data = data

        x0 = torch.randn_like(pflow)
        t = None
        t, xt, ut, _ = self.FM.sample_location_and_conditional_flow(
            x0, pflow, t=t, return_noise=True
        )
        pf_mask = mask[..., 1].unsqueeze(-1)

        vt = self.net(
            xt,
            truth,
            mask,
            timestep=t,
            global_data=global_data,
        )
        loss = self.loss(ut * pf_mask, vt * pf_mask) / pf_mask.sum() / ut.shape[-1]
        return loss

    @torch.no_grad()
    def sample(
        self,
        truth,
        pflow_shape,
        mask,
        n_steps=None,
        global_data=None,
    ):
        if n_steps is None:
            n_steps = self.n_steps
        return self.dpm.sample(
            torch.randn(pflow_shape, device=truth.device),
            truth=truth,
            mask=mask,
            global_data=global_data,
            steps=10,
            method="multistep",
            skip_type="time_uniform_flow",
        )

    def training_step(self, data, batch_idx):
        loss = self.forward(data)

        self.log(
            "train_loss",
            loss,
            batch_size=data[0].shape[0],
            sync_dist=True,
        )

        return loss

    def validation_step(self, data, batch_idx):
        loss = self.forward(data)
        truth, pflow, mask, global_data = data
        return_dict = {}
        return_dict["val_loss"] = loss
        with torch.no_grad():
            pred = self.sample(
                truth,
                pflow.shape,
                mask,
                global_data=global_data,
            )
            fs_mask = mask[..., 1].bool()
            pred_loss = self.get_pred_loss(pred, pflow, mask=mask)
            for key, val in pred_loss.items():
                self.log(
                    f"pred_{key}",
                    val.cuda(),
                    batch_size=data[0].shape[0],
                    sync_dist=True,
                )
                return_dict[f"pred_{key}"] = val
        self.log(
            "val_loss",
            loss,
            batch_size=data[0].shape[0],
            sync_dist=True,
        )

        return_dict["truth"] = truth.cpu()
        return_dict["pflow"] = pflow.cpu()
        return_dict["mask"] = mask.cpu()
        return_dict["global_data"] = global_data.cpu()
        return_dict["fs"] = pred.cpu()
        return_dict["fs_mask"] = fs_mask.cpu()

        self.validation_step_outputs.append(return_dict)

    def on_train_epoch_end(self):
        if self.config["lr_scheduler"] is not False:
            self.log("lr", self.lr_schedulers().get_last_lr()[0], sync_dist=True)
            self.lr_schedulers().step(epoch=self.current_epoch)

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss_avg", avg_loss, sync_dist=True)
        if self.current_epoch % 5 == 0:
            avg_pred_loss = torch.stack([x["pred_total_loss"] for x in outputs]).mean()
            self.log("pred_loss_avg", avg_pred_loss, sync_dist=True)
        if self.config["lr_scheduler"] is not False:
            self.log("lr", self.lr_schedulers().get_last_lr()[0], sync_dist=True)

        truth = torch.cat([x["truth"] for x in outputs], dim=0)
        pflow = torch.cat([x["pflow"] for x in outputs], dim=0)
        mask = torch.cat([x["mask"] for x in outputs], dim=0)
        global_data = torch.cat([x["global_data"] for x in outputs], dim=0)
        fs = torch.cat([x["fs"] for x in outputs], dim=0)
        fs_mask = torch.cat([x["fs_mask"] for x in outputs], dim=0)

        truth = torch.cat(
            [
                truth[..., :2],
                torch.atan2(truth[..., 2], truth[..., 3]).unsqueeze(-1) / 1.814,
                truth[..., 4:],
            ],
            dim=-1,
        )
        pflow = torch.cat(
            [
                pflow[..., :2],
                torch.atan2(pflow[..., 2], pflow[..., 3]).unsqueeze(-1) / 1.814,
                pflow[..., 4:],
            ],
            dim=-1,
        )
        fs = torch.cat(
            [
                fs[..., :2],
                torch.atan2(fs[..., 2], fs[..., 3]).unsqueeze(-1) / 1.814,
                fs[..., 4:],
            ],
            dim=-1,
        )

        val_jet_pt_mse = self.log_image(truth, pflow, mask, global_data, fs, fs_mask)
        self.log("val_jet_pt_mse", val_jet_pt_mse, sync_dist=True)

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
            default_scheduler_dict = {
                "first_cycle_steps": 10,
                "warmup_steps": 4,
                "max_lr": 4 * float(self.config["learningrate"]),
                "min_lr": 1e-5,
                "gamma": 0.8,
            }
            default_scheduler_dict.update(self.config.get("lr_scheduler_dict", {}))
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                **default_scheduler_dict,
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

    def get_pred_loss(self, pred, target, mask):
        loss = {}
        loss = self.pred_loss(pred, target, mask)
        return loss

    def log_image(self, truth, pflow, mask, global_data, fs, fs_mask=None):
        # fig, axs = plt.subplots(2, 3, figsize=(16, 8), dpi=200)#, tight_layout=True)
        n_h = 4
        n_w = 3
        fig, axs = subplots_with_absolute_sized_axes(
            n_h, n_w, figsize=(n_h * 4, 20), axis_width=4, axis_height=4, dpi=200
        )
        axs = axs.flatten()
        canvas = FigureCanvas(fig)

        truth_mask = mask[..., 0].numpy()
        pflow_mask = mask[..., 1].numpy()
        if fs_mask is None:
            fs_mask = pflow_mask
        else:
            fs_mask = fs_mask.bool().numpy()
        # pt eta phi plots

        pflow_ht_ = (
            self.var_transform_dict["ht"]
            .inverse_transform(global_data[..., -1])
            .numpy()
        )
        truth_ht_ = (
            self.var_transform_dict["ht"]
            .inverse_transform(global_data[..., -3])
            .numpy()
        )
        truth_plot_data = {}
        pflow_plot_data = {}
        fs_plot_data = {}
        for i, var in enumerate(self.pflow_variables):
            var_name = var.replace("pflow_", "")
            if var_name == "class":
                continue
            shift, scale = None, None
            truth_plot_data[var_name] = (
                self.var_transform_dict[var_name].inverse_transform(
                    truth[..., i],
                    shift=shift,
                    scale=scale,
                )
            ).numpy()
            pflow_plot_data[var_name] = (
                self.var_transform_dict[var_name].inverse_transform(
                    pflow[..., i],
                    shift=shift,
                    scale=scale,
                )
            ).numpy()
            fs_plot_data[var_name] = (
                self.var_transform_dict[var_name].inverse_transform(
                    fs[..., i],
                    shift=shift,
                    scale=scale,
                )
            ).numpy()

            if var == "phi":
                truth_plot_data[var] = normalize_phi(truth_plot_data[var])
                pflow_plot_data[var] = normalize_phi(pflow_plot_data[var])
                fs_plot_data[var] = normalize_phi(fs_plot_data[var])

        scatter_mask = fs_mask & pflow_mask
        cmap = copy.copy(plt.get_cmap("PuRd"))
        cmap.set_under("white")

        def _add_hist2d(fig, ax, pf_data, fs_data, bins):
            h, xe, ye, im = ax.hist2d(
                pf_data, fs_data, bins=bins, cmap=cmap, norm=mpl.colors.LogNorm()
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

        bins_dict = {
            "ptrel": np.linspace(0, 1, 100),
            "eta": np.linspace(-3, 3, 100),
            "phi": np.linspace(-np.pi, np.pi, 100),
            "vx": np.linspace(-2, 2, 50),
            "vy": np.linspace(-2, 2, 50),
            "vz": np.linspace(-20, 20, 50),
        }
        for i, var in enumerate(self.pflow_variables):
            var = var.replace("pflow_", "")
            if var not in bins_dict.keys():
                continue
            bins = bins_dict[var]
            _add_hist2d(
                fig,
                axs[i],
                pflow_plot_data[var][scatter_mask],
                fs_plot_data[var][scatter_mask],
                bins,
            )
            axs[i].set_title(var if var != "ptrel" else "$p_{T, rel}$")
            axs[i].set_xlabel("PFlow")
            axs[i].set_ylabel("FastSim")

        pflow_class = pflow[..., -5:].argmax(-1).numpy()[scatter_mask]
        fs_class = fs[..., -5:].argmax(-1).numpy()[scatter_mask]
        cm = confusion_matrix(pflow_class, fs_class, normalize="true")
        df_cm = pd.DataFrame(
            cm,
            index=["Ch had", "El", "Mu", "Neut had", "Phot"],
            columns=["Ch had", "El", "Mu", "Neut had", "Phot"],
        )
        sns.heatmap(
            df_cm,
            annot=True,
            annot_kws={"size": 8},
            cmap=sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True),
            cbar=False,
            ax=axs[i],
        )
        axs[i].set_title("classes")
        axs[i].set_ylabel("Pflow")
        axs[i].set_xlabel("FastSim")
        i += 1

        truth_pt = truth_plot_data["ptrel"] * truth_mask * truth_ht_.reshape(-1, 1)
        pflow_pt = pflow_plot_data["ptrel"] * pflow_mask * pflow_ht_.reshape(-1, 1)
        fs_pt = fs_plot_data["ptrel"] * fs_mask * pflow_ht_.reshape(-1, 1)

        truth_phi = truth_plot_data["phi"] * truth_mask
        pflow_phi = pflow_plot_data["phi"] * pflow_mask
        fs_phi = fs_plot_data["phi"] * fs_mask

        truth_met_x = (truth_pt * np.cos(truth_phi)).sum(-1)
        truth_met_y = (truth_pt * np.sin(truth_phi)).sum(-1)
        pflow_met_x = (pflow_pt * np.cos(pflow_phi)).sum(-1)
        pflow_met_y = (pflow_pt * np.sin(pflow_phi)).sum(-1)
        fs_met_x = (fs_pt * np.cos(fs_phi)).sum(-1)
        fs_met_y = (fs_pt * np.sin(fs_phi)).sum(-1)

        truth_ht = truth_pt.sum(-1).flatten()
        pflow_ht = pflow_pt.sum(-1).flatten()
        fs_ht = fs_pt.sum(-1).flatten()
        bins = np.linspace(0, 3000, 100)

        def _add_to_hist(ax, data, bins, label, histtype="step", **kwargs):
            ax.hist(
                data,
                bins=bins,
                histtype=histtype,
                label=f"{label} {np.mean(data):.2f}/{np.std(data):.2f}",
                **kwargs,
            )

        _add_to_hist(axs[i], pflow_ht, bins, "PFlow", histtype="stepfilled", alpha=0.5)
        _add_to_hist(axs[i], truth_ht, bins, "Truth")
        _add_to_hist(axs[i], fs_ht, bins, "FastSim")
        axs[i].set_title("Jet pT")
        axs[i].legend()

        bins = np.linspace(-200, 200, 100)
        _add_to_hist(
            axs[i + 1],
            pflow_ht - truth_ht,
            bins,
            "PFlow",
            histtype="stepfilled",
            alpha=0.5,
        )
        _add_to_hist(axs[i + 1], fs_ht - truth_ht, bins, "FastSim")
        axs[i + 1].set_title("Jet pT Residual")
        axs[i + 1].legend()

        bins = np.linspace(-100, 100, 100)
        _add_to_hist(
            axs[i + 2], pflow_met_x, bins, "PFlow", histtype="stepfilled", alpha=0.5
        )
        _add_to_hist(axs[i + 2], truth_met_x, bins, "Truth")
        _add_to_hist(axs[i + 2], fs_met_x, bins, "FastSim")
        axs[i + 2].set_title("MET x")
        axs[i + 2].legend()

        _add_to_hist(
            axs[i + 3], pflow_met_y, bins, "PFlow", histtype="stepfilled", alpha=0.5
        )
        _add_to_hist(axs[i + 3], truth_met_y, bins, "Truth")
        _add_to_hist(axs[i + 3], fs_met_y, bins, "FastSim")
        axs[i + 3].set_title("MET y")
        axs[i + 3].legend()

        class_bins = np.linspace(-0.5, 4.5, 6)
        axs[i + 4].hist(
            pflow_class,
            bins=class_bins,
            histtype="stepfilled",
            label="PFlow",
            density=True,
            alpha=0.5,
        )
        axs[i + 4].hist(
            fs_class, bins=class_bins, histtype="step", label="FastSim", density=True
        )
        axs[i + 4].set_xticks(np.arange(5), ["Ch had", "El", "Mu", "Neut had", "Phot"])
        axs[i + 4].set_title("Class")
        axs[i + 4].legend()

        canvas.draw()
        w, h = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(h), int(w), 3
        )
        for ax in axs:
            ax.set_box_aspect(1)
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
        return np.mean((fs_ht - pflow_ht) ** 2)
