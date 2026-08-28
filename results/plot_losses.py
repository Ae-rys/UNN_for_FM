import os
import glob
import math
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Config — edit this to add/remove the result folders to pull loss/error
# curves from (relative to this script, like make_grille.py). Every model
# subfolder containing a "loss.txt" (and/or "error.txt") under these
# directories is picked up automatically.
# ---------------------------------------------------------------------------
RESULT_DIRS = [
    # "results_2moons_DFB_ScCP_L1",
    # "results_2moons_DFB_ScCP_L1_small",
    # "results_2moons_DFB_ScCP_L1_K20",
    # "results_2moons_pairs",
    # "results_2moons_test_division",
    "2moons_ot_fixed_x_pred_epochs_200"
]

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


def _basename(name: str) -> str:
    """Strip the "{result_dir}/" disambiguation prefix added by _load_data,
    so categorization (startswith/in checks) operates on the model name."""
    return name.rsplit("/", 1)[-1]


def _assign_group(name: str) -> str:
    name = _basename(name)
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


def _resolve_dirs(results_dirs):
    """Accept a single dir (str) or a list of dirs; resolve each relative to
    this script (like make_grille.py) unless already absolute."""
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]
    return [d if os.path.isabs(d) else os.path.join(HERE, d) for d in results_dirs]


def _parse_metric_file(path):
    """Parse a "loss.txt" or "error.txt" file: tab/space-separated
    "epoch  value" lines, skipping any non-numeric header line."""
    epochs, values = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            try:
                epoch = int(parts[0])
                value = float(parts[1])
            except ValueError:
                continue  # header line, e.g. "epoch  w2_error"
            epochs.append(epoch)
            values.append(value)
    return epochs, values


def _parse_param_file(path):
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            params[key] = value
    return params


def _identity_key(run_dir, model_name):
    """Identity of the trained model in run_dir, used to deduplicate runs
    that appear in several RESULT_DIRS (e.g. the same MLP_baseline, or the
    same K/dual_dim point re-run in two ablation studies).

    Built from parametres.txt (model_class + K/N + dual_dim + version) when
    available, since that's what actually determines the architecture/weights
    — the folder name alone can be ambiguous across studies. Falls back to
    the bare model_name if parametres.txt is missing (older runs).
    """
    param_path = os.path.join(run_dir, "parametres.txt")
    if not os.path.exists(param_path):
        return model_name

    params = _parse_param_file(param_path)
    parts = [params.get("model_class", model_name)]
    for key in ("K", "N", "dual_dim", "version", "begin_div", "end_div", "pred"):
        if key in params:
            parts.append(f"{key}={params[key]}")
    return "_".join(parts)


def _pretty_label(run_dir, model_name):
    """Human-readable legend label built from parametres.txt
    (e.g. "DFB LFO K=10 dual=32"), falling back to the bare folder
    name if parametres.txt is missing."""
    param_path = os.path.join(run_dir, "parametres.txt")
    if not os.path.exists(param_path):
        return model_name

    params = _parse_param_file(param_path)
    model_class = params.get("model_class", model_name)
    if model_class == "small_MLP":
        return "MLP baseline"
    model_class = model_class.removesuffix("_UNN")

    bits = [model_class]
    if "version" in params:
        bits.append(params["version"])
    if "K" in params:
        bits.append(f"K={params['K']}")
    elif "N" in params:
        bits.append(f"N={params['N']}")
    if "dual_dim" in params:
        bits.append(f"dual={params['dual_dim']}")

    # Only show begin_div/end_div/pred for runs that actually come from the
    # --div-pairs study (folder name carries "_begin.../_pred..."). Every
    # DFB_UNN model has these fields with default values (False/False/"x"),
    # even plain K/dual_dim study runs that never varied them — checking the
    # folder name (rather than "value != default") avoids two problems:
    # cluttering plain K/dual_dim labels, AND silently hiding the suffix for
    # the div-study's own (False, False, "x") point, which would otherwise
    # look like it belongs to neither pred=x nor pred=v in plot_losses_by_pred.
    if "_begin" in model_name or "_pred" in model_name:
        begin_div = params.get("begin_div", "False")
        end_div   = params.get("end_div", "False")
        pred      = params.get("pred", "x")
        bits.append(f"begin={begin_div} end={end_div} pred={pred}")

    return " ".join(bits)


def _load_data(results_dirs, metric: str = "loss", max_loss: float = None, only_groups: list = None):
    """Load "{metric}.txt" (metric = "loss" or "error") from every model
    subfolder of results_dirs (a single dir or a list of dirs).

    Keys are disambiguated as "{result_dir_basename}/{model_name}" so the
    same model name appearing in several result directories (e.g. the same
    K-study point re-run in two ablation studies) doesn't collide. Runs that
    are the *same* trained model (same identity per parametres.txt) found in
    more than one result dir are only kept once (first occurrence).
    """
    dirs = _resolve_dirs(results_dirs)
    filename = f"{metric}.txt"

    metric_files = []
    for d in dirs:
        metric_files += sorted(glob.glob(os.path.join(d, "**", filename), recursive=True))
    if not metric_files:
        raise FileNotFoundError(f"No {filename} files found in {results_dirs}")

    data = {}
    seen_identities = set()
    for path in metric_files:
        run_dir = os.path.dirname(path)
        model_name = os.path.basename(run_dir)
        dir_label = os.path.basename(os.path.dirname(run_dir))
        key = f"{dir_label}/{model_name}"

        identity = _identity_key(run_dir, model_name)
        if identity in seen_identities:
            continue

        epochs, values = _parse_metric_file(path)
        if epochs and (max_loss is None or values[-1] < max_loss):
            if only_groups is None or _assign_group(model_name) in only_groups:
                params_path = os.path.join(run_dir, "params.txt")
                n_params = None
                if os.path.exists(params_path):
                    with open(params_path) as f:
                        n_params = int(f.read().strip())
                label = _pretty_label(run_dir, model_name)
                data[key] = (epochs, values, n_params, label)
                seen_identities.add(identity)

    if not data:
        raise ValueError("No model matches the given filters")

    # ignore first value because it's often too big and messes up the plots
    for key in data:
        epochs, values, n_params, label = data[key]
        data[key] = (epochs[1:], values[1:], n_params, label)

    return data


def _save_or_show(fig, output_path):
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def _result_name(results_dirs) -> str:
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]
    return "+".join(os.path.basename(os.path.normpath(d)) for d in results_dirs)


_METRIC_LABELS = {"loss": "Loss", "error": "W2 error"}


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


def plot_all_losses(results_dir, output_path: str = None, figsize=(14, 11),
                     max_loss: float = None, metric: str = "loss", ylim_top: float = None):
    """
    Plots all "{metric}.txt" files found under results_dir (a single folder,
    or a list of folders — like make_grille.py's RESULT_DIRS).
    metric: "loss" (training FM loss) or "error" (W2 distance to target).
    Models are colored by group (l1 / shared / no-conv / conv / baseline);
    linestyle and marker vary within each group so individual curves stay readable.
    If max_loss is set, only models whose final value is below that threshold are plotted.
    ylim_top: if set, caps the y-axis at this value (curves that start much
    higher get clipped) so the bulk of the models stay readable.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss)
    metric_label = _METRIC_LABELS[metric]

    group_counters = {g: 0 for g in _GROUP_COLORS}
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, (epochs, values, n_params, label) in data.items():
        group = _assign_group(model_name)
        color = _GROUP_COLORS[group]
        ls, mk = _style_for(group_counters[group])
        group_counters[group] += 1
        ax.plot(epochs, values, label=model_name, color=color,
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
                ordered_labels.append(_display_label(data[l][3], data[l][2]))

    ax.legend(ordered_handles, ordered_labels,
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, borderaxespad=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top, bottom=0)
    title = f"Training {metric_label.lower()} — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"
    if ylim_top is not None:
        title += f" [y capped at {ylim_top}]"
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
    name = _basename(name)
    if "LFO" in name:
        return _LFO_COLOR
    if "LNO" in name:
        return _LNO_COLOR
    return _NONE_COLOR


def plot_losses_by_group(results_dir, output_path: str = None,
                         figsize=(14, 11), max_loss: float = None, metric: str = "loss",
                         ylim_top: float = None):
    """
    Single graph (all groups combined). Color = LFO (red) vs LNO (blue);
    baselines are purple. Linestyle and marker vary within the same color to
    tell individual models apart.
    metric: "loss" or "error" (W2 distance to target).
    ylim_top: if set, caps the y-axis at this value.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss)
    metric_label = _METRIC_LABELS[metric]

    fig, ax = plt.subplots(figsize=figsize)
    color_counters = {_LFO_COLOR: 0, _LNO_COLOR: 0, _NONE_COLOR: 0}

    for name, (epochs, values, n_params, label) in data.items():
        color = _lfo_lno_color(name)
        ls, mk = _style_for(color_counters[color])
        color_counters[color] += 1
        ax.plot(epochs, values, label=name, color=color,
                linestyle=ls, marker=mk, linewidth=1.5, markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top, bottom=0)
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
                ordered_l.append(_display_label(data[l][3], data[l][2]))

    ax.legend(ordered_h, ordered_l, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=6.5, borderaxespad=0)

    title = f"Training {metric_label.lower()} by group — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"
    if ylim_top is not None:
        title += f" [y capped at {ylim_top}]"
    ax.set_title(title, fontweight="bold")
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
    name = _basename(name)
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


def _plot_by_algo(ax, data, title, legend_outside=False, legend_fontsize=6.5,
                   metric_label="Loss", ylim_top=None):
    algo_counters = {a: 0 for a in _ALGO_COLORS}

    for name, (epochs, values, n_params, label) in data.items():
        algo  = _assign_algo(name)
        color = _ALGO_COLORS[algo]
        ls, mk = _style_for(algo_counters[algo])
        algo_counters[algo] += 1
        ax.plot(epochs, values, label=name, color=color,
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
                ordered_l.append(_display_label(data[l][3], data[l][2]))

    if legend_outside:
        ax.legend(ordered_h, ordered_l, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, borderaxespad=0)
    else:
        ax.legend(ordered_h, ordered_l, fontsize=legend_fontsize, loc="upper right")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top, bottom=0)
    ax.grid(True, alpha=0.3)


def plot_losses_by_algo(results_dir, output_path: str = None,
                        figsize=(14, 11), max_loss: float = None,
                        only_groups: list = None, metric: str = "loss",
                        ylim_top: float = None):
    """
    Single graph. Color = algorithm family (CP / ScCP / DFB / DiFB / baseline).
    Linestyle and marker vary within the same color to tell individual models apart.
    metric: "loss" or "error" (W2 distance to target).
    only_groups: if set, restrict to models whose _assign_group() is in this list.
                 e.g. only_groups=["conv", "baseline"]
    ylim_top: if set, caps the y-axis at this value.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss, only_groups=only_groups)
    metric_label = _METRIC_LABELS[metric]

    title = f"Training {metric_label.lower()} by algorithm — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"
    if ylim_top is not None:
        title += f" [y capped at {ylim_top}]"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_by_algo(ax, data, title, legend_outside=True, metric_label=metric_label, ylim_top=ylim_top)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


def _plot_split(ax, data, assign_fn, colors, section_titles, title,
               legend_outside=False, legend_fontsize=6.5, metric_label="Loss"):
    counters = {c: 0 for c in colors}

    for name, (epochs, values, n_params, label) in data.items():
        cat   = assign_fn(name)
        color = colors[cat]
        ls, mk = _style_for(counters[cat])
        counters[cat] += 1
        ax.plot(epochs, values, label=name, color=color,
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
                ordered_l.append(_display_label(data[l][3], data[l][2]))

    if legend_outside:
        ax.legend(ordered_h, ordered_l, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, borderaxespad=0)
    else:
        ax.legend(ordered_h, ordered_l, fontsize=legend_fontsize, loc="upper right")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
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
    name = _basename(name)
    if "baseline" in name:
        return "baseline"
    return "L1" if "L1" in name else "non_L1"


def plot_losses_l1_vs_non_l1(results_dir, output_path: str = None,
                             figsize=(16, 8), max_loss: float = None, metric: str = "loss"):
    """
    Single graph. Color = L1 prox (red) vs non-L1 (blue); baselines stay purple.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss)
    metric_label = _METRIC_LABELS[metric]

    title = f"L1 vs non-L1 prox — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_split(ax, data, _assign_l1_category, _L1_COLORS, _L1_TITLES, title,
                legend_outside=True, metric_label=metric_label)
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
    name = _basename(name)
    if "baseline" in name:
        return "baseline"
    return "shared" if name.startswith("Shared") else "non_shared"


def plot_losses_shared_vs_non_shared(results_dir, output_path: str = None,
                                     figsize=(16, 8), max_loss: float = None, metric: str = "loss"):
    """
    Single graph. Color = Shared (blue) vs non-shared (green); baselines stay purple.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss)
    metric_label = _METRIC_LABELS[metric]

    title = f"Shared vs non-shared — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_split(ax, data, _assign_shared_category, _SHARED_COLORS, _SHARED_TITLES, title,
                legend_outside=True, metric_label=metric_label)
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
    name = _basename(name)
    if "baseline" in name:
        return "baseline"
    return "conv" if "Conv" in name else "non_conv"


def plot_losses_conv_vs_non_conv(results_dir, output_path: str = None,
                                 figsize=(16, 8), max_loss: float = None, metric: str = "loss"):
    """
    Single graph. Color = Conv (orange) vs non-conv (cyan); baselines stay purple.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss)
    metric_label = _METRIC_LABELS[metric]

    title = f"Conv vs non-conv — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_split(ax, data, _assign_conv_category, _CONV_COLORS, _CONV_TITLES, title,
                legend_outside=True, metric_label=metric_label)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


# ---- pred="x" vs pred="v" (DFB_UNN begin_div/end_div/pred division study) ----

_PREDX_COLOR = "#d62728"  # red
_PREDV_COLOR = "#1f77b4"  # blue

_PRED_COLORS = {"predx": _PREDX_COLOR, "predv": _PREDV_COLOR}
_PRED_TITLES = {"predx": "— pred=x —", "predv": "— pred=v —"}


def _assign_pred_category(name: str):
    """Returns "predx"/"predv" for runs from run_2moons.py's --div-pairs
    study (folders named "..._predx"/"..._predv"), or None otherwise."""
    name = _basename(name)
    if "predv" in name:
        return "predv"
    if "predx" in name:
        return "predx"
    return None


def plot_losses_by_pred(results_dir, output_path: str = None,
                        figsize=(14, 11), max_loss: float = None, metric: str = "loss",
                        ylim_top: float = None):
    """
    Single graph restricted to the begin_div/end_div/pred division-study
    runs (see run_2moons.py --div-pairs). Color = pred="x" (red) vs
    pred="v" (blue); everything else (regular K/dual_dim studies, ScCP,
    baselines) is excluded since "pred" doesn't apply to them.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss)
    metric_label = _METRIC_LABELS[metric]

    data = {k: v for k, v in data.items() if _assign_pred_category(k) is not None}
    if not data:
        raise ValueError("No pred=x / pred=v division-study runs found")

    title = f"Training {metric_label.lower()} by pred — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"
    if ylim_top is not None:
        title += f" [y capped at {ylim_top}]"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_split(ax, data, _assign_pred_category, _PRED_COLORS, _PRED_TITLES, title,
                legend_outside=True, metric_label=metric_label)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top, bottom=0)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


# ---- begin_div=False vs begin_div=True (DFB_UNN division study) ----

_BEGINF_COLOR = "#d62728"  # red   — begin_div=False
_BEGINT_COLOR = "#1f77b4"  # blue  — begin_div=True

_BEGIN_DIV_COLORS = {"beginF": _BEGINF_COLOR, "beginT": _BEGINT_COLOR}
_BEGIN_DIV_TITLES = {"beginF": "— begin_div=False —", "beginT": "— begin_div=True —"}


def _assign_begin_div_category(name: str):
    """Returns "beginF"/"beginT" for runs from run_2moons.py's --div-pairs
    study (folders named "..._beginFalse.../..._beginTrue..."), or None otherwise."""
    name = _basename(name)
    if "beginTrue" in name:
        return "beginT"
    if "beginFalse" in name:
        return "beginF"
    return None


def plot_losses_by_begin_div(results_dir, output_path: str = None,
                             figsize=(14, 11), max_loss: float = None, metric: str = "loss",
                             ylim_top: float = None):
    """
    Single graph restricted to the begin_div/end_div/pred division-study
    runs (see run_2moons.py --div-pairs). Color = begin_div=False (red) vs
    begin_div=True (blue); everything else is excluded.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss)
    metric_label = _METRIC_LABELS[metric]

    data = {k: v for k, v in data.items() if _assign_begin_div_category(k) is not None}
    if not data:
        raise ValueError("No begin_div division-study runs found")

    title = f"Training {metric_label.lower()} by begin_div — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"
    if ylim_top is not None:
        title += f" [y capped at {ylim_top}]"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_split(ax, data, _assign_begin_div_category, _BEGIN_DIV_COLORS, _BEGIN_DIV_TITLES, title,
                legend_outside=True, metric_label=metric_label)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top, bottom=0)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


# ---- end_div=False vs end_div=True (DFB_UNN division study) ----

_ENDF_COLOR = "#d62728"  # red   — end_div=False
_ENDT_COLOR = "#1f77b4"  # blue  — end_div=True

_END_DIV_COLORS = {"endF": _ENDF_COLOR, "endT": _ENDT_COLOR}
_END_DIV_TITLES = {"endF": "— end_div=False —", "endT": "— end_div=True —"}


def _assign_end_div_category(name: str):
    """Returns "endF"/"endT" for runs from run_2moons.py's --div-pairs
    study (folders named "..._endFalse.../..._endTrue..."), or None otherwise."""
    name = _basename(name)
    if "endTrue" in name:
        return "endT"
    if "endFalse" in name:
        return "endF"
    return None


def plot_losses_by_end_div(results_dir, output_path: str = None,
                           figsize=(14, 11), max_loss: float = None, metric: str = "loss",
                           ylim_top: float = None):
    """
    Single graph restricted to the begin_div/end_div/pred division-study
    runs (see run_2moons.py --div-pairs). Color = end_div=False (red) vs
    end_div=True (blue); everything else is excluded.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss)
    metric_label = _METRIC_LABELS[metric]

    data = {k: v for k, v in data.items() if _assign_end_div_category(k) is not None}
    if not data:
        raise ValueError("No end_div division-study runs found")

    title = f"Training {metric_label.lower()} by end_div — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"
    if ylim_top is not None:
        title += f" [y capped at {ylim_top}]"

    fig, ax = plt.subplots(figsize=figsize)
    _plot_split(ax, data, _assign_end_div_category, _END_DIV_COLORS, _END_DIV_TITLES, title,
                legend_outside=True, metric_label=metric_label)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top, bottom=0)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


# ---- LFO vs LNO, split into two graphs, color-coded by K ----

_K_COLORMAP = "viridis"


def _get_param_value(key, field, default=None):
    """Re-reads parametres.txt for a _load_data key ("{result_dir}/{model_name}")
    and returns `field` (or `default` if missing / no parametres.txt)."""
    run_dir = os.path.join(HERE, key)
    param_path = os.path.join(run_dir, "parametres.txt")
    if not os.path.exists(param_path):
        return default
    params = _parse_param_file(param_path)
    return params.get(field, default)


def plot_losses_by_K(results_dir, version, output_path: str = None,
                     figsize=(14, 11), max_loss: float = None, metric: str = "loss",
                     ylim_top: float = None):
    """
    Single graph restricted to runs matching `version` ("LFO" or "LNO"),
    plus the MLP baseline. Color = number of layers K — one discrete color
    per distinct K value (curves with different dual_dim but the same K
    share a color); linestyle/marker vary within the same color so
    individual curves stay distinguishable. Full legend: one entry per
    curve (exact model label), grouped under a "— K=... —" header per color.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss)
    metric_label = _METRIC_LABELS[metric]

    filtered = {
        k: v for k, v in data.items()
        if "baseline" in _basename(k) or version in _basename(k)
    }
    if not filtered:
        raise ValueError(f"No {version} runs (or baseline) found")

    k_values = sorted({
        int(_get_param_value(key, "K") or _get_param_value(key, "N"))
        for key in filtered
        if "baseline" not in _basename(key)
        and (_get_param_value(key, "K") or _get_param_value(key, "N")) is not None
    })
    if not k_values:
        raise ValueError(f"No K values found among {version} runs")

    cmap = plt.colormaps[_K_COLORMAP]
    color_for_K = {
        K: cmap(i / max(len(k_values) - 1, 1)) for i, K in enumerate(k_values)
    }

    fig, ax = plt.subplots(figsize=figsize)
    style_counters = {"baseline": 0, **{K: 0 for K in k_values}}
    entries_by_group = {"baseline": [], **{K: [] for K in k_values}}

    for key, (epochs, values, n_params, label) in filtered.items():
        if "baseline" in _basename(key):
            group, color = "baseline", _BASELINE_COLOR
        else:
            group = int(_get_param_value(key, "K") or _get_param_value(key, "N"))
            color = color_for_K[group]
        ls, mk = _style_for(style_counters[group])
        style_counters[group] += 1
        h, = ax.plot(epochs, values, color=color, linestyle=ls, marker=mk,
                     linewidth=1.5, markersize=4)
        entries_by_group[group].append((h, _display_label(label, n_params)))

    ordered_h, ordered_l = [], []
    for group in ["baseline"] + k_values:
        if entries_by_group[group]:
            ordered_h.append(plt.Line2D([], [], color="none"))
            ordered_l.append("— baseline —" if group == "baseline" else f"— K={group} —")
            for h, l in entries_by_group[group]:
                ordered_h.append(h)
                ordered_l.append(l)

    ax.legend(ordered_h, ordered_l, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, borderaxespad=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_label)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top, bottom=0)

    title = f"Training {metric_label.lower()} — {version} — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"
    if ylim_top is not None:
        title += f" [y capped at {ylim_top}]"
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


# ---- Combined overview ----

def plot_losses_overview(results_dir, output_path: str = None,
                         figsize=(20, 14), max_loss: float = None, metric: str = "loss"):
    """
    2x2 overview: L1 vs non-L1 | Shared vs non-shared
                   By algorithm | Conv vs non-conv
    Baselines are always purple.
    """
    data = _load_data(results_dir, metric=metric, max_loss=max_loss)
    metric_label = _METRIC_LABELS[metric]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    _plot_split(axes[0, 0], data, _assign_l1_category, _L1_COLORS, _L1_TITLES,
                "L1 vs non-L1", metric_label=metric_label)
    _plot_split(axes[0, 1], data, _assign_shared_category, _SHARED_COLORS, _SHARED_TITLES,
                "Shared vs non-shared", metric_label=metric_label)
    _plot_by_algo(axes[1, 0], data, "By algorithm", metric_label=metric_label)

    title = f"Training {metric_label.lower()} overview — {_result_name(results_dir)}"
    if max_loss is not None:
        title += f" (final {metric_label.lower()} < {max_loss})"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    _save_or_show(fig, output_path)
    return fig


# Some models' training loss starts very high (especially the unstable
# begin_div/end_div=True DFB runs), which crushes the y-axis and hides the
# detail for every other model. LOSS_YLIM_CAP adds a 4th, capped, version of
# each "losses_*.png" plot so the well-behaved curves stay readable.
LOSS_YLIM_CAP = 0.2


def make_plots(results_dirs=None, out_dir=None):
    """Generate the loss AND W2-error versions of the main comparison plots
    for every model found under `results_dirs` (defaults to RESULT_DIRS,
    edit that list at the top of this file to add/remove folders). Also
    generates a y-capped (<= LOSS_YLIM_CAP) version of each "losses_*.png" plot.
    """
    results_dirs = results_dirs if results_dirs is not None else RESULT_DIRS
    out_dir = out_dir if out_dir is not None else HERE

    for metric, prefix in [("loss", "losses"), ("error", "errors")]:
        try:
            plot_all_losses(results_dirs, metric=metric,
                             output_path=os.path.join(out_dir, f"{prefix}_comparison.png"))
            plot_losses_by_algo(results_dirs, metric=metric,
                                 output_path=os.path.join(out_dir, f"{prefix}_by_algo.png"))
            plot_losses_by_group(results_dirs, metric=metric,
                                  output_path=os.path.join(out_dir, f"{prefix}_by_group.png"))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [skip {metric}] {exc}")

        try:
            plot_losses_by_pred(results_dirs, metric=metric,
                                 output_path=os.path.join(out_dir, f"{prefix}_by_pred.png"))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [skip {metric} by_pred] {exc}")

        try:
            plot_losses_by_begin_div(results_dirs, metric=metric,
                                      output_path=os.path.join(out_dir, f"{prefix}_by_begin_div.png"))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [skip {metric} by_begin_div] {exc}")

        try:
            plot_losses_by_end_div(results_dirs, metric=metric,
                                    output_path=os.path.join(out_dir, f"{prefix}_by_end_div.png"))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [skip {metric} by_end_div] {exc}")

        for version in ["LFO", "LNO"]:
            try:
                plot_losses_by_K(results_dirs, version, metric=metric,
                                  output_path=os.path.join(out_dir, f"{prefix}_{version}_by_K.png"))
            except (FileNotFoundError, ValueError) as exc:
                print(f"  [skip {metric} {version} by_K] {exc}")

    try:
        plot_all_losses(results_dirs, metric="loss", ylim_top=LOSS_YLIM_CAP,
                         output_path=os.path.join(out_dir, "losses_comparison_capped.png"))
        plot_losses_by_algo(results_dirs, metric="loss", ylim_top=LOSS_YLIM_CAP,
                             output_path=os.path.join(out_dir, "losses_by_algo_capped.png"))
        plot_losses_by_group(results_dirs, metric="loss", ylim_top=LOSS_YLIM_CAP,
                              output_path=os.path.join(out_dir, "losses_by_group_capped.png"))
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [skip capped loss] {exc}")

    try:
        plot_losses_by_pred(results_dirs, metric="loss", ylim_top=LOSS_YLIM_CAP,
                             output_path=os.path.join(out_dir, "losses_by_pred_capped.png"))
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [skip capped loss by_pred] {exc}")

    try:
        plot_losses_by_begin_div(results_dirs, metric="loss", ylim_top=LOSS_YLIM_CAP,
                                  output_path=os.path.join(out_dir, "losses_by_begin_div_capped.png"))
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [skip capped loss by_begin_div] {exc}")

    try:
        plot_losses_by_end_div(results_dirs, metric="loss", ylim_top=LOSS_YLIM_CAP,
                                output_path=os.path.join(out_dir, "losses_by_end_div_capped.png"))
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [skip capped loss by_end_div] {exc}")

    for version in ["LFO", "LNO"]:
        try:
            plot_losses_by_K(results_dirs, version, metric="loss", ylim_top=LOSS_YLIM_CAP,
                              output_path=os.path.join(out_dir, f"losses_{version}_by_K_capped.png"))
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [skip capped loss {version} by_K] {exc}")


if __name__ == "__main__":
    make_plots()
