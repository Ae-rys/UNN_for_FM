import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# One color per group; linestyle+marker combos distinguish models within a group.
_GROUP_COLORS = {
    "shared":   "#1f77b4",  # blue
    "no_conv":  "#2ca02c",  # green
    "conv":     "#ff7f0e",  # orange
    "baseline": "#9467bd",  # purple
}
_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS    = ["o", "s", "^", "D", "v", "p", "*", "X"]


def _assign_group(name: str) -> str:
    if name.startswith("Shared"):
        return "shared"
    if "Conv" not in name and "UNet" not in name:
        return "no_conv"
    if "UNet" in name:
        return "baseline"
    return "conv"


def _style_for(index: int):
    ls = _LINESTYLES[index // len(_MARKERS) % len(_LINESTYLES)]
    mk = _MARKERS[index % len(_MARKERS)]
    return ls, mk


def plot_all_losses(results_dir: str, output_path: str = None, figsize=(16, 8), max_loss: float = None):
    """
    Plots all loss.txt files found in results_dir subdirectories.
    Models are colored by group (shared / no-conv / conv / baseline);
    linestyle and marker vary within each group so individual curves stay readable.
    If max_loss is set, only models whose final loss is below that threshold are plotted.
    """
    loss_files = sorted(glob.glob(os.path.join(results_dir, "**", "loss.txt"), recursive=True))
    if not loss_files:
        raise FileNotFoundError(f"No loss.txt files found in {results_dir}")

    data = {}
    for path in loss_files:
        model_name = os.path.basename(os.path.dirname(path))
        epochs, losses = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    epochs.append(int(parts[0]))
                    losses.append(float(parts[1]))
        if epochs and (max_loss is None or losses[-1] < max_loss):
            data[model_name] = (epochs, losses)

    if not data:
        raise ValueError(f"No model has a final loss below {max_loss}")

    # Count per group to cycle styles independently
    group_counters = {g: 0 for g in _GROUP_COLORS}

    fig, ax = plt.subplots(figsize=figsize)

    for model_name, (epochs, losses) in data.items():
        group = _assign_group(model_name)
        color = _GROUP_COLORS[group]
        ls, mk = _style_for(group_counters[group])
        group_counters[group] += 1
        ax.plot(epochs, losses, label=model_name, color=color,
                linestyle=ls, marker=mk, linewidth=1.5, markersize=4)

    # Group legend entries with a blank separator between groups
    handles, labels = ax.get_legend_handles_labels()
    grouped = {"shared": [], "no_conv": [], "conv": [], "baseline": []}
    for h, l in zip(handles, labels):
        grouped[_assign_group(l)].append((h, l))

    ordered_handles, ordered_labels = [], []
    group_titles = {"shared": "— Shared —", "no_conv": "— No conv —",
                    "conv": "— Conv —", "baseline": "— Baselines —"}
    for g, title in group_titles.items():
        if grouped[g]:
            ordered_handles.append(plt.Line2D([], [], color="none"))
            ordered_labels.append(title)
            for h, l in grouped[g]:
                ordered_handles.append(h)
                ordered_labels.append(l)

    ax.legend(ordered_handles, ordered_labels,
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, borderaxespad=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    title = "Training loss — results_regular_version"
    if max_loss is not None:
        title += f" (final loss < {max_loss})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    return fig


_LFO_COLOR  = "#d62728"   # red
_LNO_COLOR  = "#1f77b4"   # blue
_NONE_COLOR = "#7f7f7f"   # grey (baselines with no LFO/LNO tag)

_GROUP_LABELS = {
    "shared":   "Shared",
    "no_conv":  "No conv",
    "conv":     "Conv (non-shared)",
    "baseline": "Baselines (UNet)",
}


def _lfo_lno_color(name: str) -> str:
    if "LFO" in name:
        return _LFO_COLOR
    if "LNO" in name:
        return _LNO_COLOR
    return _NONE_COLOR


def plot_losses_by_group(results_dir: str, output_path: str = None,
                         figsize=(18, 12), max_loss: float = None):
    """
    2x2 subplot grid, one panel per group (shared / no-conv / conv / baseline).
    Within each panel, color = LFO (red) vs LNO (blue).
    Linestyle and marker vary within the same color to tell individual models apart.
    """
    loss_files = sorted(glob.glob(os.path.join(results_dir, "**", "loss.txt"), recursive=True))
    if not loss_files:
        raise FileNotFoundError(f"No loss.txt files found in {results_dir}")

    data = {}
    for path in loss_files:
        model_name = os.path.basename(os.path.dirname(path))
        epochs, losses = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    epochs.append(int(parts[0]))
                    losses.append(float(parts[1]))
        if epochs and (max_loss is None or losses[-1] < max_loss):
            data[model_name] = (epochs, losses)

    if not data:
        raise ValueError(f"No model has a final loss below {max_loss}")

    groups = {g: {} for g in _GROUP_LABELS}
    for name, series in data.items():
        groups[_assign_group(name)][name] = series

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, (group_key, group_label) in zip(axes, _GROUP_LABELS.items()):
        group_data = groups[group_key]
        color_counters = {_LFO_COLOR: 0, _LNO_COLOR: 0, _NONE_COLOR: 0}

        for name, (epochs, losses) in group_data.items():
            color = _lfo_lno_color(name)
            ls, mk = _style_for(color_counters[color])
            color_counters[color] += 1
            ax.plot(epochs, losses, label=name, color=color,
                    linestyle=ls, marker=mk, linewidth=1.5, markersize=4)

        ax.set_title(group_label, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        lfo   = [(h, l) for h, l in zip(handles, labels) if "LFO" in l]
        lno   = [(h, l) for h, l in zip(handles, labels) if "LNO" in l]
        other = [(h, l) for h, l in zip(handles, labels) if "LFO" not in l and "LNO" not in l]

        ordered_h, ordered_l = [], []
        for section, title in [(lfo, "LFO"), (lno, "LNO"), (other, "")]:
            if section:
                if title:
                    ordered_h.append(plt.Line2D([], [], color="none"))
                    ordered_l.append(f"— {title} —")
                for h, l in section:
                    ordered_h.append(h)
                    ordered_l.append(l)

        ax.legend(ordered_h, ordered_l, fontsize=6.5, loc="upper right")

    title = "Training loss by group — results_regular_version"
    if max_loss is not None:
        title += f" (final loss < {max_loss})"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    return fig


_ALGO_COLORS = {
    "CP":       "#1f77b4",  # blue
    "ScCP":     "#2ca02c",  # green
    "DFB":      "#ff7f0e",  # orange
    "baseline": "#9467bd",  # purple
}


def _assign_algo(name: str) -> str:
    if "ScCP" in name:
        return "ScCP"
    if "CP" in name:
        return "CP"
    if "DFB" in name:
        return "DFB"
    return "baseline"


def plot_losses_by_algo(results_dir: str, output_path: str = None,
                        figsize=(16, 8), max_loss: float = None,
                        only_groups: list = None):
    """
    Single graph. Color = algorithm family (CP / ScCP / DFB / baseline).
    Linestyle and marker vary within the same color to tell individual models apart.
    only_groups: if set, restrict to models whose _assign_group() is in this list.
                 e.g. only_groups=["conv", "baseline"]
    """
    loss_files = sorted(glob.glob(os.path.join(results_dir, "**", "loss.txt"), recursive=True))
    if not loss_files:
        raise FileNotFoundError(f"No loss.txt files found in {results_dir}")

    data = {}
    for path in loss_files:
        model_name = os.path.basename(os.path.dirname(path))
        epochs, losses = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    epochs.append(int(parts[0]))
                    losses.append(float(parts[1]))
        if epochs and (max_loss is None or losses[-1] < max_loss):
            if only_groups is None or _assign_group(model_name) in only_groups:
                data[model_name] = (epochs, losses)

    if not data:
        raise ValueError(f"No model matches the given filters")

    algo_counters = {a: 0 for a in _ALGO_COLORS}
    fig, ax = plt.subplots(figsize=figsize)

    for name, (epochs, losses) in data.items():
        algo  = _assign_algo(name)
        color = _ALGO_COLORS[algo]
        ls, mk = _style_for(algo_counters[algo])
        algo_counters[algo] += 1
        ax.plot(epochs, losses, label=name, color=color,
                linestyle=ls, marker=mk, linewidth=1.5, markersize=4)

    handles, labels = ax.get_legend_handles_labels()
    grouped = {a: [] for a in _ALGO_COLORS}
    for h, l in zip(handles, labels):
        grouped[_assign_algo(l)].append((h, l))

    ordered_h, ordered_l = [], []
    for algo in _ALGO_COLORS:
        if grouped[algo]:
            ordered_h.append(plt.Line2D([], [], color="none"))
            ordered_l.append(f"— {algo} —")
            for h, l in grouped[algo]:
                ordered_h.append(h)
                ordered_l.append(l)

    ax.legend(ordered_h, ordered_l,
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, borderaxespad=0)

    title = "Training loss by algorithm — results_regular_version"
    if max_loss is not None:
        title += f" (final loss < {max_loss})"
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "results_OT_only_prox_conv")
    plot_all_losses(results_dir, output_path=os.path.join(results_dir, "losses_comparison.png"))
    plot_all_losses(
        results_dir,
        output_path=os.path.join(results_dir, "losses_comparison_below0.5.png"),
        max_loss=0.5,
    )
    plot_losses_by_group(results_dir, output_path=os.path.join(results_dir, "losses_by_group.png"))
    plot_losses_by_algo(results_dir, output_path=os.path.join(results_dir, "losses_by_algo.png"))
    plot_losses_by_algo(
        results_dir,
        output_path=os.path.join(results_dir, "losses_by_algo_conv_baselines.png"),
        only_groups=["conv", "baseline", "shared"],
    )
