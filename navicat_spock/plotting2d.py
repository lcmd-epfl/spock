#!/usr/bin/env python

import matplotlib
import numpy as np
import scipy.stats as stats
import matplotlib
import math
import os

if os.name == "posix" and "DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from navicat_spock.exceptions import MissingDataError
from navicat_spock.helpers import bround, namefixer


def calc_ci(resid, n, dof, x, x2, y2):
    t = stats.t.ppf(0.95, dof)
    s_err = np.sqrt(np.sum(resid**2) / dof)

    ci = (
        t
        * s_err
        * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )

    return ci


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
    assert len(px) == len(py) == len(cb) == len(ms)
    if plotmode == 1:
        s = 15
        lw = 0.25
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
    fig,
    ax,
    xmin=0,
    xmax=100,
    xbase=20,
    ybase=10,
    xlabel="X-axis",
    ylabel="Y-axis",
    filename="plot.png",
    cb="white",
    ms="o",
    breakpoints=None,
    estimates=None,
    plotmode=1,
    save_fig=True,
):
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
    elif plotmode == 1:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95, zorder=1)
        ax = beautify_ax(ax)
        plotpoints(ax, px, py, cb, ms, plotmode)
    elif plotmode == 2:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95, zorder=1)
        for bp in breakpoints:
            ax.axvline(
                bp,
                linestyle="dotted",
                linewidth=0.75,
                alpha=0.75,
                color="lightsteelblue",
            )
        ax = beautify_ax(ax)
        plotpoints(ax, px, py, cb, ms, plotmode)
    elif plotmode == 3:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95, zorder=1)
        for bp in breakpoints:
            ax.axvline(
                bp,
                linestyle="dotted",
                linewidth=0.75,
                alpha=0.75,
                color="lightsteelblue",
            )
        ax = beautify_ax(ax)
        plotpoints(ax, px, py, cb, ms, plotmode)
        for bp_i in range(len(breakpoints)):
            bp_ci = estimates["breakpoint{}".format(bp_i + 1)]["confidence_interval"]
            plt.axvspan(bp_ci[0], bp_ci[1], alpha=0.1, color="#b9cfe7")
    ymin, ymax = ax.get_ylim()
    ymax = bround(ymax, ybase, type="max")
    ymin = bround(ymin, ybase, type="min")
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(ymin, ymax + 0.1, ybase))
    if save_fig:
        plt.savefig(filename)
    if os.name != "posix" and "DISPLAY" in os.environ:
        plt.show()
    return fig, ax


def plot_and_save(
    pw_fit,
    tags,
    idx,
    tidx,
    cb,
    ms,
    plotmode,
    fig,
    ax,
    return_value=True,
    save_fig=True,
    save_csv=True,
):
    # Try to figure out good dimensions for the axes and ticks
    x = pw_fit.xx
    y = pw_fit.yy
    xint = np.linspace(min(pw_fit.xx), max(pw_fit.xx), 250)
    xspread = np.abs(max(pw_fit.xx) - min(pw_fit.xx))
    yspread = np.abs(max(pw_fit.yy) - min(pw_fit.yy))
    xom = 10 ** (np.floor(math.log(xspread, 10)))
    yom = 10 ** (np.floor(math.log(yspread, 10)))
    xbase = bround(xspread / 10, xom, type="max")
    ybase = bround(yspread / 10, yom, type="max")

    # Getting the raw data to plot
    final_params = pw_fit.best_muggeo.best_fit.raw_params
    breakpoints = pw_fit.best_muggeo.best_fit.next_breakpoints
    estimates = pw_fit.best_muggeo.best_fit.estimates

    # Extract what we need from params
    intercept_hat = final_params[0]
    alpha_hat = final_params[1]
    beta_hats = final_params[2 : 2 + len(breakpoints)]

    # Build the fit plot segment by segment
    yint = intercept_hat + alpha_hat * xint
    for bp_count in range(len(breakpoints)):
        yint += beta_hats[bp_count] * np.maximum(xint - breakpoints[bp_count], 0)
    fig, ax = plot_2d(
        xint,
        yint,
        x,
        y,
        fig,
        ax,
        xmin=min(pw_fit.xx),
        xmax=max(pw_fit.xx),
        xbase=xbase,
        ybase=ybase,
        xlabel=tags[idx],
        ylabel=tags[tidx],
        cb=cb,
        ms=ms,
        breakpoints=breakpoints,
        estimates=estimates,
        plotmode=plotmode,
        filename=f"{namefixer(tags[idx].strip())}_volcano.png",
        save_fig=save_fig,
    )

    # Pass in standard matplotlib keywords to control any of the plots
    # pw_fit.plot_breakpoint_confidence_intervals()

    # Print to file
    if save_csv:
        zdata = list(zip(xint, yint))
        csvname = f"{namefixer(tags[idx].strip())}_volcano.csv"
        np.savetxt(
            csvname, zdata, fmt="%.4e", delimiter=",", header="{tags[idx]},{tags[tidx]}"
        )
    if return_value:
        return fig, ax
    else:
        plt.close()
        return None
