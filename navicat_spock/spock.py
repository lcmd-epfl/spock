#!/usr/bin/env python

import sys
import traceback
import numpy as np
import matplotlib.pyplot as plt

from navicat_spock.helpers import (
    processargs,
    group_data_points,
    curate_d,
    bround,
    namefixer,
    reweighter,
)
from navicat_spock.exceptions import InputError, ConvergenceError
from navicat_spock.piecewise_regression import ModelSelection, Fit
from navicat_spock.plotting2d import plot_and_save


def run_spock():
    (df, verb, imputer_strat, plotmode) = processargs(sys.argv[1:])
    run_spock_from_args(df, verb, imputer_strat, plotmode)


def run_spock_from_args(df, verb=0, imputer_strat="none", plotmode=1):
    prefit = False
    if verb > 0:
        print(
            f"spock will assume that {df.columns[0]} contains names/IDs of catalysts/samples."
        )
    names = df[df.columns[0]].values

    # Atttempts to group data points based on shared characters in names.
    cb, ms = group_data_points(0, 2, names)

    # Expecting a full reaction profile with corresponding column names. Descriptors will be identified.
    tags = np.array([str(tag) for tag in df.columns[1:]], dtype=object)
    d = np.float64(df.to_numpy()[:, 1:])

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

    # Target data
    target = d[:, tidx]  # .reshape(-1)
    weights = reweighter(target)
    if verb > 4:
        print("Weights for the regression of the target are:")
        for y, w in zip(target, weights):
            print(y, w)
    idxs = np.where(descriptors == True)[0]
    best_bics = np.zeros_like(idxs, dtype=float)
    best_n = np.zeros_like(idxs, dtype=int)
    for i, idx in enumerate(idxs):
        try:
            if verb > 0:
                print(f"Attempting fit with descriptor index {idx}: {tags[idx]}...:")
            descriptor = d[:, idx].reshape(-1)
            msel = ModelSelection(
                descriptor,
                target,
                max_breakpoints=3,
                max_iterations=250,
                weights=weights,
            )
            bic_list = np.array(
                [summary["bic"] for summary in msel.model_summaries], dtype=float
            )
            n_list = np.array(
                [summary["n_breakpoints"] for summary in msel.model_summaries],
                dtype=int,
            )
            filter_nan = ~np.isnan(bic_list)
            bic_list = bic_list[filter_nan]
            n_list = n_list[filter_nan]
            n = int(n_list[np.argmin(bic_list)])
            best_bics[i] = np.min(bic_list)
            best_n[i] = n
            if verb > 2:
                print(f"The best number of breakpoints according to BIC is {n}")
            if n < 1 and verb > 0:
                print(
                    f"The best number of breakpoints is less than 1 because the algorithm could not converge. Exiting!"
                )
                raise ConvergenceError(
                    "Algorithm did not converge for any number of breaking points."
                )
            if prefit and verb > 1:
                # Fit piecewise regression!
                pw_fit = Fit(
                    descriptor,
                    target,
                    n_breakpoints=n,
                    weights=weights,
                    max_iterations=250,
                )
                if verb > 2:
                    pw_fit.summary()
                if not pw_fit.best_muggeo:
                    raise ConvergenceError("The fitting process did not converge.")
                if verb > 1:
                    # Plot the data, fit, breakpoints and confidence intervals
                    _ = plot_and_save(pw_fit, tags, idx, tidx, cb, ms, plotmode)
        except Exception as m:
            traceback.print_exc()
            best_bics[i] = np.inf
            if verb > 0:
                print(
                    f"Fit did not converge with descriptor index {idx}: {tags[idx]}\n due to {m}"
                )
    if any(best_bics != 0):
        idx = idxs[np.argmin(best_bics)]
        n = int(best_n[np.argmin(best_bics)])
        if verb > 3:
            print(f"Attempting fit with {n} breakpoints as determined from BIC")
        descriptor = d[:, idx].reshape(-1)
        pw_fit = Fit(
            descriptor, target, n_breakpoints=n, weights=weights, max_iterations=250
        )
        if not pw_fit.best_muggeo:
            raise ConvergenceError("The fitting process did not converge.")
        if verb > 2:
            pw_fit.summary()
        # Plot the data, fit, breakpoints and confidence intervals
        fig = plot_and_save(pw_fit, tags, idx, tidx, cb, ms, plotmode)
        plt.show()


if __name__ == "__main__":
    run_spock()
