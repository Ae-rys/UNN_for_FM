import os
import glob
import math
import matplotlib.pyplot as plt
import numpy as np

_BASELINE_COLOR = "#9467bd"  # purple — always used for baselines (MLP/UNet), everywhere

# One color per group; linestyle+marker combos distinguish models within a group.
_GROUP_COLORS = {
    "l1":       "#d62728",  # red
    "shared":   "#1f77b4",  # blue
    "no_conv":  "#2ca02c",  # green
    "conv":     "#ff7f0e",  # orange
    "baseline": _BASELINE_COLOR,
}
_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS    = ["o", "s", "^", "D", "v", "p", "*", "X"]


def _assign_group(name: str) -> str:
    if "baseline" in name:
        return "baseline"
    if "L1" in name:
        return "l1"
    if name.startswith("Shared"):
        return "shared"
    if "Conv" in name:
        return "conv"
    return "no_conv"


def _style_for(index: int):
    ls = _LINESTYLES[index // len(_MARKERS) % len(_LINESTYLES)]
    mk = _MARKERS[index % len(_MARKERS)]
    return ls, mk


def _load_data(results_dir: str, max_loss: float = None, only_groups: list = None):
    loss_files = sorted(glob.glob(os.path.join(results_dir, "**", "loss.txt"), recursive=True))
    if not loss_files:
        raise FileNotFoundError(f"No loss.txt files found in {results_dir}")

    data = {}
    for path in loss_files:
        run_dir = os.path.dirname(path)
        model_name = os.path.basename(run_dir)
        epochs, losses = [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    epochs.append(int(parts[0]))
                    losses.append(float(parts[1]))
        if epochs and (max_loss is None or losses[-1] < max_loss):
            if only_groups is None or _assign_group(model_name) in only_groups:
                params_path = os.path.join(run_dir, "params.txt")
                n_params = None
                if os.path.exists(params_path):
                    with open(params_path) as f:
                        n_params = int(f.read().strip())
                data[model_name] = (epochs, losses, n_params)

    if not data:
        raise ValueError("No model matches the given filters")
    
    #ignore first value in data because the loss is too big and messes up the plots
    for model_name in data:
        epochs, losses, n_params = data[model_name]
        data[model_name] = (epochs[1:], losses[1:], n_params)
        
    return data


def _save_or_show(fig, output_path):
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def _result_name(results_dir: str) -> str:
    return os.path.basename(os.path.normpath(results_dir))


def _format_params(n_params: int) -> str:
    if n_params is None:
        return ""
    if n_params >= 1_000_000:
        return f"{n_params / 1_000_000:.2f}M"
    if n_params >= 1_000:
        return f"{n_params / 1_000:.1f}K"
    return str(n_params)


def _display_label(name: str, n_params: int) -> str:
    if n_params is None:
        return name
    return f"{name} ({_format_params(n_params)})"


def plot_all_losses(results_dir: str, output_path: str = None, figsize=(16, 8), max_loss: float = None):
    """
    Plots all loss.txt files found in results_dir subdirectories.
    Models are colored by group (l1 / shared / no-conv / conv / baseline);
    linestyle and marker vary within each group so individual curves stay readable.
    If max_loss is set, only models whose final loss is below that threshold are plotted.
    """
    data = _load_data(results_dir, max_loss)

    group_counters = {g: 0 for g in _GROUP_COLORS}
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, (epochs, losses, n_params) in data.items():
        group = _assign_group(model_name)
        color = _GROUP_COLORS[group]
        ls, mk = _style_for(group_counters[group])
        group_counters[group] += 1
        ax.plot(epochs, losses, label=model_name, color=color,
                linestyle=ls, marker=mk, linewidth=1.5, markersize=4)

    # Group legend entries with a blank separator between groups
    handles, labels = ax.get_legend_handles_labels()
    grouped = {g: [] for g in _GROUP_COLORS}
    for h, l in zip(handles, labels):
        grouped[_assign_group(l)].append((h, l))

    ordered_handles, ordered_labels = [], []
    group_titles = {"l1": "— L1 —", "shared": "— Shared —", "no_conv": "— No conv —",
                    "conv": "— Conv —", "baseline": "— Baselines —"}
    for g, title in group_titles.items():
        if grouped[g]:
            ordered_handles.append(plt.Line2D([], [], color="none"))
            ordered_labels.append(title)
            for h, l in grouped[g]:
                ordered_handles.append(h)
                ordered_labels.append(_display_label(l, data[l][2]))

    ax.legend(ordered_handles, ordered_labels,
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, borderaxespad=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    title = f"Training loss — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final loss < {max_loss})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


_LFO_COLOR  = "#d62728"   # red
_LNO_COLOR  = "#1f77b4"   # blue
_NONE_COLOR = _BASELINE_COLOR  # baselines (no LFO/LNO tag) are always purple

_GROUP_LABELS = {
    "l1":       "L1 prox",
    "shared":   "Shared",
    "no_conv":  "No conv",
    "conv":     "Conv (non-shared)",
    "baseline": "Baselines",
}


def _lfo_lno_color(name: str) -> str:
    if "LFO" in name:
        return _LFO_COLOR
    if "LNO" in name:
        return _LNO_COLOR
    return _NONE_COLOR


def plot_losses_by_group(results_dir: str, output_path: str = None,
                         figsize=None, max_loss: float = None):
    """
    Grid of subplots, one panel per non-empty group (l1 / shared / no-conv / conv / baseline).
    Within each panel, color = LFO (red) vs LNO (blue); baselines are purple.
    Linestyle and marker vary within the same color to tell individual models apart.
    """
    data = _load_data(results_dir, max_loss)

    groups = {g: {} for g in _GROUP_LABELS}
    for name, series in data.items():
        groups[_assign_group(name)][name] = series

    present = [(g, l) for g, l in _GROUP_LABELS.items() if groups[g]]
    n = len(present)
    ncols = 3 if n > 4 else 2
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")

    for ax, (group_key, group_label) in zip(axes, present):
        group_data = groups[group_key]
        color_counters = {_LFO_COLOR: 0, _LNO_COLOR: 0, _NONE_COLOR: 0}

        for name, (epochs, losses, n_params) in group_data.items():
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
        for section, sec_title in [(lfo, "LFO"), (lno, "LNO"), (other, "")]:
            if section:
                if sec_title:
                    ordered_h.append(plt.Line2D([], [], color="none"))
                    ordered_l.append(f"— {sec_title} —")
                for h, l in section:
                    ordered_h.append(h)
                    ordered_l.append(_display_label(l, group_data[l][2]))

        ax.legend(ordered_h, ordered_l, fontsize=6.5, loc="upper right")

    title = f"Training loss by group — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final loss < {max_loss})"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


_ALGO_COLORS = {
    "CP":       "#1f77b4",  # blue
    "ScCP":     "#2ca02c",  # green
    "DFB":      "#ff7f0e",  # orange
    "DiFB":     "#8c564b",  # brown
    "baseline": _BASELINE_COLOR,
}


def _assign_algo(name: str) -> str:
    if "baseline" in name:
        return "baseline"
    if "ScCP" in name:
        return "ScCP"
    if "CP" in name:
        return "CP"
    if "DiFB" in name:
        return "DiFB"
    if "DFB" in name:
        return "DFB"
    return "baseline"


def _plot_by_algo(ax, data, title, legend_outside=False, legend_fontsize=6.5):
    algo_counters = {a: 0 for a in _ALGO_COLORS}

    for name, (epochs, losses, n_params) in data.items():
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
                ordered_l.append(_display_label(l, data[l][2]))

    if legend_outside:
        ax.legend(ordered_h, ordered_l, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, borderaxespad=0)
    else:
        ax.legend(ordered_h, ordered_l, fontsize=legend_fontsize, loc="upper right")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)


def plot_losses_by_algo(results_dir: str, output_path: str = None,
                        figsize=(16, 8), max_loss: float = None,
                        only_groups: list = None):
    """
    Single graph. Color = algorithm family (CP / ScCP / DFB / DiFB / baseline).
    Linestyle and marker vary within the same color to tell individual models apart.
    only_groups: if set, restrict to models whose _assign_group() is in this list.
                 e.g. only_groups=["conv", "baseline"]
    """
    data = _load_data(results_dir, max_loss, only_groups=only_groups)

    title = f"Training loss by algorithm — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final loss < {max_loss})"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_by_algo(ax, data, title, legend_outside=True)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


def _plot_split(ax, data, assign_fn, colors, section_titles, title,
               legend_outside=False, legend_fontsize=6.5):
    counters = {c: 0 for c in colors}

    for name, (epochs, losses, n_params) in data.items():
        cat   = assign_fn(name)
        color = colors[cat]
        ls, mk = _style_for(counters[cat])
        counters[cat] += 1
        ax.plot(epochs, losses, label=name, color=color,
                linestyle=ls, marker=mk, linewidth=1.5, markersize=4)

    handles, labels = ax.get_legend_handles_labels()
    grouped = {c: [] for c in colors}
    for h, l in zip(handles, labels):
        grouped[assign_fn(l)].append((h, l))

    ordered_h, ordered_l = [], []
    for cat, sec_title in section_titles.items():
        if grouped[cat]:
            ordered_h.append(plt.Line2D([], [], color="none"))
            ordered_l.append(sec_title)
            for h, l in grouped[cat]:
                ordered_h.append(h)
                ordered_l.append(_display_label(l, data[l][2]))

    if legend_outside:
        ax.legend(ordered_h, ordered_l, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, borderaxespad=0)
    else:
        ax.legend(ordered_h, ordered_l, fontsize=legend_fontsize, loc="upper right")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)


# ---- L1 vs non-L1 ----

_L1_COLOR     = "#d62728"  # red
_NON_L1_COLOR = "#1f77b4"  # blue

_L1_COLORS = {
    "L1":       _L1_COLOR,
    "non_L1":   _NON_L1_COLOR,
    "baseline": _BASELINE_COLOR,
}
_L1_TITLES = {"L1": "— L1 —", "non_L1": "— non-L1 —", "baseline": "— Baseline —"}


def _assign_l1_category(name: str) -> str:
    if "baseline" in name:
        return "baseline"
    return "L1" if "L1" in name else "non_L1"


def plot_losses_l1_vs_non_l1(results_dir: str, output_path: str = None,
                             figsize=(16, 8), max_loss: float = None):
    """
    Single graph. Color = L1 prox (red) vs non-L1 (blue); baselines stay purple.
    """
    data = _load_data(results_dir, max_loss)

    title = f"L1 vs non-L1 prox — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final loss < {max_loss})"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_split(ax, data, _assign_l1_category, _L1_COLORS, _L1_TITLES, title, legend_outside=True)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


# ---- Shared vs non-shared ----

_SHARED_COLOR     = "#1f77b4"  # blue
_NON_SHARED_COLOR = "#2ca02c"  # green

_SHARED_COLORS = {
    "shared":     _SHARED_COLOR,
    "non_shared": _NON_SHARED_COLOR,
    "baseline":   _BASELINE_COLOR,
}
_SHARED_TITLES = {"shared": "— Shared —", "non_shared": "— Non-shared —", "baseline": "— Baseline —"}


def _assign_shared_category(name: str) -> str:
    if "baseline" in name:
        return "baseline"
    return "shared" if name.startswith("Shared") else "non_shared"


def plot_losses_shared_vs_non_shared(results_dir: str, output_path: str = None,
                                     figsize=(16, 8), max_loss: float = None):
    """
    Single graph. Color = Shared (blue) vs non-shared (green); baselines stay purple.
    """
    data = _load_data(results_dir, max_loss)

    title = f"Shared vs non-shared — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final loss < {max_loss})"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_split(ax, data, _assign_shared_category, _SHARED_COLORS, _SHARED_TITLES, title, legend_outside=True)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


# ---- Conv vs non-conv ----

_CONV_COLOR     = "#ff7f0e"  # orange
_NON_CONV_COLOR = "#17becf"  # cyan

_CONV_COLORS = {
    "conv":     _CONV_COLOR,
    "non_conv": _NON_CONV_COLOR,
    "baseline": _BASELINE_COLOR,
}
_CONV_TITLES = {"conv": "— Conv —", "non_conv": "— Non-conv —", "baseline": "— Baseline —"}


def _assign_conv_category(name: str) -> str:
    if "baseline" in name:
        return "baseline"
    return "conv" if "Conv" in name else "non_conv"


def plot_losses_conv_vs_non_conv(results_dir: str, output_path: str = None,
                                 figsize=(16, 8), max_loss: float = None):
    """
    Single graph. Color = Conv (orange) vs non-conv (cyan); baselines stay purple.
    """
    data = _load_data(results_dir, max_loss)

    title = f"Conv vs non-conv — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final loss < {max_loss})"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_split(ax, data, _assign_conv_category, _CONV_COLORS, _CONV_TITLES, title, legend_outside=True)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


# ---- Combined overview ----

def plot_losses_overview(results_dir: str, output_path: str = None,
                         figsize=(20, 14), max_loss: float = None):
    """
    2x2 overview: L1 vs non-L1 | Shared vs non-shared
                   By algorithm | Conv vs non-conv
    Baselines are always purple.
    """
    data = _load_data(results_dir, max_loss)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    _plot_split(axes[0, 0], data, _assign_l1_category, _L1_COLORS, _L1_TITLES, "L1 vs non-L1")
    _plot_split(axes[0, 1], data, _assign_shared_category, _SHARED_COLORS, _SHARED_TITLES, "Shared vs non-shared")
    _plot_by_algo(axes[1, 0], data, "By algorithm")

    title = f"Training loss overview — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final loss < {max_loss})"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "2_moons_only_l1")
    plot_all_losses(results_dir, output_path=os.path.join(results_dir, "losses_comparison.png"))
    plot_losses_by_group(results_dir, output_path=os.path.join(results_dir, "losses_by_group.png"))
    plot_losses_by_algo(results_dir, output_path=os.path.join(results_dir, "losses_by_algo.png"))
    plot_losses_l1_vs_non_l1(results_dir, output_path=os.path.join(results_dir, "losses_l1_vs_non_l1.png"))
    plot_losses_shared_vs_non_shared(results_dir, output_path=os.path.join(results_dir, "losses_shared_vs_non_shared.png"))
    plot_losses_conv_vs_non_conv(results_dir, output_path=os.path.join(results_dir, "losses_conv_vs_non_conv.png"))
    plot_losses_overview(results_dir, output_path=os.path.join(results_dir, "losses_overview.png"))
