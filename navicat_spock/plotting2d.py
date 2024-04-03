#!/usr/bin/env python

import matplotlib
import numpy as np
import scipy.stats as stats
import matplotlib
import os

if os.name == "posix" and "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from navicat_spock.exceptions import MissingDataError
from navicat_spock.helpers import bround, namefixer


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
    if os.name != "posix" and "DISPLAY" in os.environ:
        plt.show()
    return fig


def plot_and_save(pw_fit, tags, idx, tidx, cb, ms, plotmode):
    # fig = plot_and_save(pw_fit, tags, idx, tidx)
    x = pw_fit.xx
    y = pw_fit.yy
    xint = np.linspace(min(pw_fit.xx), max(pw_fit.xx), 250)
    xbase = bround(np.abs(max(pw_fit.xx) - min(pw_fit.xx)) / 10, type="max")
    ybase = bround(np.abs(max(pw_fit.yy) - min(pw_fit.yy)) / 10, type="max")
    final_params = pw_fit.best_muggeo.best_fit.raw_params
    breakpoints = pw_fit.best_muggeo.best_fit.next_breakpoints

    # Extract what we need from params
    intercept_hat = final_params[0]
    alpha_hat = final_params[1]
    beta_hats = final_params[2 : 2 + len(breakpoints)]

    # Build the fit plot segment by segment
    yint = intercept_hat + alpha_hat * xint
    for bp_count in range(len(breakpoints)):
        yint += beta_hats[bp_count] * np.maximum(xint - breakpoints[bp_count], 0)
    fig = plot_2d(
        xint,
        yint,
        x,
        y,
        xmin=min(pw_fit.xx),
        xmax=max(pw_fit.xx),
        xbase=xbase,
        ybase=ybase,
        xlabel=tags[idx],
        ylabel=tags[tidx],
        cb=cb,
        ms=ms,
        plotmode=plotmode,
        filename=f"{namefixer(tags[idx].strip())}_volcano.png",
    )

    # Pass in standard matplotlib keywords to control any of the plots
    # pw_fit.plot_breakpoints()
    # pw_fit.plot_breakpoint_confidence_intervals()

    # Print to file
    zdata = list(zip(xint, yint))
    csvname = f"{namefixer(tags[idx].strip())}_volcano.csv"
    np.savetxt(
        csvname, zdata, fmt="%.4e", delimiter=",", header="{tags[idx]},{tags[tidx]}"
    )
    return fig
