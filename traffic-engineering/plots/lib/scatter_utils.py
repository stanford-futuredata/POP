import matplotlib.pyplot as plt
import seaborn as sns
from .plot_utils import save_figure, COLOR_NAMES_DICT


def scatter_plot(
    ratio_dfs,
    labels,
    short_labels,
    x_axis,
    y_axis,
    xlabel=None,
    ylabel=None,
    xlim=0,
    ylim=(10e-3, 5e3),
    title=None,
    xlog=False,
    ylog=False,
    bbta=(0, 0, 1, 1),
    ncol=2,
    figsize=(8, 6),
    ax=None,
    show_legend=True,
    arrow_coords=None,
    show=False,
    save=True,
):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for ratio_df, label, short_label in zip(ratio_dfs, labels, short_labels):
        ax.scatter(
            ratio_df[x_axis],
            ratio_df[y_axis],
            alpha=0.75,
            label=label,
            marker="o",
            linewidth=1,
            color=COLOR_NAMES_DICT[short_label],
        )
    if xlim:
        ax.set_xlim(0)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    extra_artists = []
    if arrow_coords:
        bbox_props = {
            "boxstyle": "rarrow,pad=0.5",
            "fc": "white",
            "ec": "black",
            "lw": 2,
        }
        t = ax.text(
            arrow_coords[0],
            arrow_coords[1],
            "Better",
            ha="center",
            va="center",
            rotation=45,
            size=16,
            color="black",
            bbox=bbox_props,
        )
        extra_artists.append(t)

    if show_legend:
        legend = ax.legend(
            loc="center",
            bbox_to_anchor=bbta,
            ncol=ncol,
            frameon=False,
            handletextpad=0.2,
            columnspacing=0.2,
        )
        extra_artists.append(legend)
    sns.despine()
    if show:
        plt.show()
    if save:
        save_figure(
            "scatter-plot-{}-{}-{}".format(x_axis, y_axis, title),
            extra_artists=extra_artists,
        )
