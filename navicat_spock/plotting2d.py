#!/usr/bin/env python

import matplotlib
import numpy as np
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model

from navicat_spock.exceptions import MissingDataError
from navicat_spock.helpers import bround


def calc_ci(resid, n, dof, x, x2, y2):
    t = stats.t.ppf(0.95, dof)
    s_err = np.sqrt(np.sum(resid ** 2) / dof)

    ci = (
        t
        * s_err
        * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )

    return ci


def plot_ci(ci, x2, y2, ax=None):
    if ax is None:
        try:
            ax = plt.gca()
        except Exception as m:
            return

    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", alpha=0.6)

    return ax


def beautify_ax(ax):
    # Border
    ax.spines["top"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.get_xaxis().set_tick_params(direction="out")
    ax.get_yaxis().set_tick_params(direction="out")
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    return ax


def plotpoints(ax, px, py, cb, ms, plotmode):
    if plotmode == 1:
        s = 30
        lw = 0.3
    else:
        s = 15
        lw = 0.25
    for i in range(len(px)):
        ax.scatter(
            px[i],
            py[i],
            s=s,
            c=cb[i],
            marker=ms[i],
            linewidths=lw,
            edgecolors="black",
            zorder=2,
        )


def plot_2d(
    x,
    y,
    px,
    py,
    ci=None,
    xmin=0,
    xmax=100,
    xbase=20,
    ybase=10,
    xlabel="X-axis",
    ylabel="Y-axis",
    filename="plot.png",
    rid=None,
    rb=None,
    cb="white",
    ms="o",
    plotmode=1,
):
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )
    # Labels and key
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    xmax = bround(xmax, xbase, type="max")
    xmin = bround(xmin, xbase, type="min")
    plt.xlim(xmin, xmax)
    plt.xticks(np.arange(xmin, xmax + 0.1, xbase))
    if plotmode == 0:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95)
        ax = beautify_ax(ax)
        if rid is not None and rb is not None:
            avgs = []
            rb.append(xmax)
            for i in range(len(rb) - 1):
                avgs.append((rb[i] + rb[i + 1]) / 2)
            for i in rb:
                ax.axvline(
                    i, linestyle="dashed", color="black", linewidth=0.75, alpha=0.75
                )
    elif plotmode == 1:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95, zorder=1)
        ax = beautify_ax(ax)
        if rid is not None and rb is not None:
            avgs = []
            rb.append(xmax)
            for i in range(len(rb) - 1):
                avgs.append((rb[i] + rb[i + 1]) / 2)
            for i in rb:
                ax.axvline(
                    i,
                    linestyle="dashed",
                    color="black",
                    linewidth=0.75,
                    alpha=0.75,
                    zorder=3,
                )
        plotpoints(ax, px, py, cb, ms, plotmode)
    elif plotmode == 2:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95, zorder=1)
        ax = beautify_ax(ax)
        if rid is not None and rb is not None:
            avgs = []
            rb.append(xmax)
            for i in range(len(rb) - 1):
                avgs.append((rb[i] + rb[i + 1]) / 2)
            for i in rb:
                ax.axvline(
                    i,
                    linestyle="dashed",
                    color="black",
                    linewidth=0.5,
                    alpha=0.75,
                    zorder=3,
                )
            yavg = (y.max() + y.min()) * 0.5
            for i, j in zip(rid, avgs):
                plt.text(
                    j,
                    yavg,
                    i,
                    fontsize=7.5,
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation="vertical",
                    zorder=4,
                )
        plotpoints(ax, px, py, cb, ms, plotmode)
    elif plotmode == 3:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95, zorder=1)
        ax = beautify_ax(ax)
        if rid is not None and rb is not None:
            avgs = []
            rb.append(xmax)
            for i in range(len(rb) - 1):
                avgs.append((rb[i] + rb[i + 1]) / 2)
            for i in rb:
                ax.axvline(
                    i,
                    linestyle="dashed",
                    color="black",
                    linewidth=0.75,
                    alpha=0.75,
                    zorder=3,
                )
        plotpoints(ax, px, py, cb, ms, plotmode)
        if ci is not None:
            plot_ci(ci, x, y, ax=ax)
    ymin, ymax = ax.get_ylim()
    ymax = bround(ymax, ybase, type="max")
    ymin = bround(ymin, ybase, type="min")
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(ymin, ymax + 0.1, ybase))
    plt.savefig(filename)
    matplotlib.use("QtAgg")
    return fig
