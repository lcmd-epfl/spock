#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

from navicat_spock.helpers import (
    processargs,
    group_data_points,
    curate_d,
    bround,
    namefixer,
)
from navicat_spock.exceptions import InputError, ConvergenceError
from navicat_spock.piecewise_regression import ModelSelection, Fit
from navicat_spock.plotting2d import plot_2d


def run_spock():
    (df, verb, imputer_strat, plotmode) = processargs(sys.argv[1:])

    if verb > 0:
        print(
            f"spock will assume that {df.columns[0]} contains names/IDs of catalysts/samples."
        )
    names = df[df.columns[0]].values

    # Atttempts to group data points based on shared characters in names.
    cb, ms = group_data_points(0, 2, names)

    # Expecting a full reaction profile with corresponding column names. Descriptors will be identified.
    tags = np.array([str(tag) for tag in df.columns[1:]], dtype=object)
    d = np.float32(df.to_numpy()[:, 1:])

    # TS or intermediate are interpreted from column names. Coeffs is a boolean array.
    descriptors = np.zeros(len(tags), dtype=bool)
    for i, tag in enumerate(tags):
        if "TARGET" in tag.upper():
            if verb > 0:
                print(f"Assuming field {tag} corresponds to the TARGET (y-axis).")
            tidx = i
        else:
            if verb > 0:
                print(
                    f"Assuming field {tag} corresponds to a possible descriptor variable."
                )
            descriptors[i] = True

    # Your data might contain outliers (human error, computation error) or missing points.
    # We will attempt to curate your data automatically.
    d, cb, ms, names = curate_d(d, descriptors, cb, ms, names, imputer_strat, verb=verb)
    idxs = np.where(descriptors == True)[0]
    best_bics = np.zeros_like(idxs)
    for i, idx in enumerate(idxs):
        if verb > 0:
            print(f"Attempting fit with descriptor index {idx}: {tags[idx]}...:")
        descriptor = d[:, idx].reshape(-1)
        target = d[:, tidx]  # .reshape(-1)

        # Weights
        norm = sum(target)  # Not needed sin robust regression will normalize
        weights = np.array(
            [py ** 4 for py in target]
        )  # **2 at least, could be increased
        for y, w in zip(target, weights):
            # print(y,w)
            continue
        msel = ModelSelection(
            descriptor, target, max_breakpoints=3, max_iterations=1000, weights=weights
        )
        bic_list = np.array(
            [summary["bic"] for summary in msel.model_summaries], dtype=float
        )
        n_list = np.array(
            [summary["n_breakpoints"] for summary in msel.model_summaries], dtype=int
        )
        filter_nan = ~np.isnan(bic_list)
        bic_list = bic_list[filter_nan]
        n_list = n_list[filter_nan]
        n = int(n_list[np.argmin(bic_list)])
        best_bics[i] = np.min(bic_list)
        if verb > 0:
            print(f"The best number of breakpoints according to BIC is {n}")
        if n < 1 and verb > 0:
            print(
                f"The best number of breakpoints is less than 1 because the algorithm could not converge. Exiting!"
            )
            raise ConvergenceError(
                "Algorithm did not converge for any number of breaking points."
            )

        # Fit piecewise regression!
        pw_fit = Fit(descriptor, target, n_breakpoints=n, weights=weights)
        if verb > 2:
            pw_fit.summary()

        # Plot the data, fit, breakpoints and confidence intervals
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

        if verb > 1:
            # Build the fit plot segment by segment
            yint = intercept_hat + alpha_hat * xint
            for bp_count in range(len(breakpoints)):
                yint += beta_hats[bp_count] * np.maximum(
                    xint - breakpoints[bp_count], 0
                )
            if not pw_fit.best_muggeo:
                raise ConvergenceError("The fitting process did not converge.")
            plot_2d(
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
                csvname,
                zdata,
                fmt="%.4e",
                delimiter=",",
                header="{tags[idx]},{tags[tidx]}",
            )
    if any(best_bics != 0):
        idx = idxs[np.argmin(best_bics)]
        descriptor = d[:, idx].reshape(-1)
        target = d[:, tidx]  # .reshape(-1)
        # Weights
        norm = sum(target)  # Not needed sin robust regression will normalize
        weights = np.array(
            [py ** 4 for py in target]
        )  # **2 at least, could be increased
        pw_fit = Fit(descriptor, target, n_breakpoints=n, weights=weights)
        if verb > 2:
            pw_fit.summary()
        # Plot the data, fit, breakpoints and confidence intervals
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
        if not pw_fit.best_muggeo:
            raise ConvergenceError("The fitting process did not converge.")
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
        plt.show()
