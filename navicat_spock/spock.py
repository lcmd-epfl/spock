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
    n_iter_helper,
    slope_check,
)
from navicat_spock.exceptions import InputError, ConvergenceError
from navicat_spock.piecewise_regression import ModelSelection, Fit
from navicat_spock.plotting2d import plot_and_save


def run_spock():
    (df, wp, verb, imputer_strat, plotmode) = processargs(sys.argv[1:])
    _ = run_spock_from_args(df, wp, verb, imputer_strat, plotmode)


def run_spock_from_args(df, wp=2, verb=0, imputer_strat="none", plotmode=1):
    prefit = False
    fitted = False
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
                print(
                    f"Assuming field {tag} corresponds to the TARGET (y-axis), which will be weighted with power {wp}."
                )
            tidx = i
        else:
            if verb > 0:
                print(
                    f"Assuming field {tag} corresponds to a possible descriptor variable."
                )
            descriptors[i] = True

    # Your data might contain outliers (human error, computation error) or missing points.
    # We will attempt to curate your data automatically.
    try:
        d, cb, ms, names = curate_d(
            d, descriptors, cb, ms, names, imputer_strat, verb=verb
        )
    except Exception as m:
        pass

    # Target data
    target = d[:, tidx]  # .reshape(-1)
    weights = reweighter(target, wp)
    if verb > 4:
        print("Weights for the regression of the target are:")
        for y, w in zip(target, weights):
            print(y, w)
    max_breakpoints = 2
    idxs = np.where(descriptors == True)[0]
    all_bic = np.zeros((len(idxs), max_breakpoints + 1), dtype=float)
    all_n = np.zeros((len(idxs), max_breakpoints + 1), dtype=int)
    all_sc = np.zeros((len(idxs), max_breakpoints + 1), dtype=bool)
    for i, idx in enumerate(idxs):
        try:
            if verb > 0:
                print(f"Attempting fit with descriptor index {idx}: {tags[idx]}...:")
            descriptor = d[:, idx].reshape(-1)
            xrange = 0.05 * (max(descriptor) - min(descriptor))
            msel = ModelSelection(
                descriptor,
                target,
                max_breakpoints=max_breakpoints,
                max_iterations=n_iter_helper(fitted),
                weights=weights,
                tolerance=xrange,
                verbose=verb > 2,
            )
            all_sc[i, :] = np.array(
                [slope_check(summary["betas"]) for summary in msel.model_summaries],
                dtype=bool,
            )
            all_bic[i, :] = np.array(
                [summary["bic"] for summary in msel.model_summaries], dtype=float
            )
            all_n[i, :] = np.array(
                [summary["n_breakpoints"] for summary in msel.model_summaries],
                dtype=int,
            )

            bic_list = all_bic[i, :]
            n_list = all_n[i, :]
            n = n_list[np.argmin(bic_list)]
            min_bic = np.min(bic_list)

            if verb > 3:
                print(
                    f"The list of BICs for the n breakpoints are:\n {bic_list} for\n {n_list}"
                )
            if verb > 2:
                print(f"The best number of breakpoints according to BIC is {n}")
            if n < 1:
                if verb > 1:
                    print(
                        f"BIC seems to indicate that a linear fit is better than a volcano fit. Warning!"
                    )
                    if verb > 3:
                        print(
                            f"Adding {n} as the best BIC for this descriptor.\nAdding {min_bic} as the best BIC for this descriptor."
                        )
            else:
                if verb > 3:
                    print(
                        f"After zero removal, the list of BICs for the n breakpoints are:\n {bic_list} for\n {n_list}"
                    )
                if any(bic_list):
                    if n > 0:
                        if verb > 3:
                            print(
                                f"Adding {n} as the best BIC for this descriptor.\nAdding {min_bic} as the best BIC for this descriptor."
                            )
                        fitted = True
            if prefit and verb > 1:
                # Fit piecewise regression!
                pw_fit = Fit(
                    descriptor,
                    target,
                    n_breakpoints=n,
                    weights=weights,
                    max_iterations=n_iter_helper(False),
                    tolerance=xrange,
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
            all_bic[i, :] = np.inf
            all_n[i, :] = 0
            all_sc[i, :] = False
            if verb > 0:
                print(
                    f"Fit did not converge with descriptor index {idx}: {tags[idx]}\n due to {m}"
                )

    # Done iterating over descriptors
    best_n = np.zeros_like(idxs)
    best_bic = np.zeros_like(idxs)
    best_sc = np.zeros_like(idxs)
    for i, idx in enumerate(idxs):

        bic_list = all_bic[i, :]
        n_list = all_n[i, :]
        sc_list = all_sc[i, :]

        filter_nan = ~np.isnan(bic_list)
        bic_list = bic_list[filter_nan]
        n_list = n_list[filter_nan]
        sc_list = sc_list[filter_nan]

        if any(sc_list):
            bic_list = bic_list[sc_list]
            n_list = n_list[sc_list]
        elif any(n_list):
            filter_0s = np.nonzero(n_list)
            bic_list = bic_list[filter_0s]
            n_list = n_list[filter_0s]

        # Save best current n and bic
        n = n_list[np.argmin(bic_list)]
        min_bic = np.min(bic_list)
        best_n[i] = n
        best_bic[i] = min_bic

    filter_0s = np.nonzero(best_n)
    best_bic_nz = best_bic[filter_0s]
    best_n_nz = best_n[filter_0s]

    if verb > 3 and any(best_n_nz):
        print(
            f"Out of all descriptors, the list of BICs for the n>0 breakpoints are:\n {best_bic} for\n {best_n}"
        )
    if any(best_bic):
        if any(best_n_nz):
            min_bic = np.min(best_bic_nz)
            n = int(best_n[np.where(best_bic == min_bic)[0][0]])
            idx = idxs[np.where(best_bic == min_bic)[0][0]]
            if verb > 3:
                print(
                    f"Removing n=0 solutions, {n} breakpoints for index {idx}: {tags[idx]} will be used."
                )
            if verb > 1:
                print(
                    f"Fitting volcano with {n} breakpoints and descriptor index {idx}: {tags[idx]}, as determined from BIC."
                )
            descriptor = d[:, idx].reshape(-1)
            xrange = 0.05 * (max(descriptor) - min(descriptor))
            pw_fit = Fit(
                descriptor,
                target,
                n_breakpoints=n,
                weights=weights,
                max_iterations=5000,
                tolerance=xrange,
            )
            if not pw_fit.best_muggeo:
                raise ConvergenceError("The fitting process did not converge.")
            if verb > 2:
                pw_fit.summary()
            # Plot the data, fit, breakpoints and confidence intervals
            fig = plot_and_save(pw_fit, tags, idx, tidx, cb, ms, plotmode)
            return fig
        else:
            min_bic = np.min(best_bic)
            idx = idxs[np.argmin(best_bic)]
            n = int(best_n[np.argmin(best_bic)])
            if verb > 3:
                print(
                    f"Considering n=0 solutions, {n} breakpoints for index {idx}: {tags[idx]} should be used. This does not correspond to a volcano. Exiting!"
                )
            exit()
    else:
        print("None of the descriptors could be fit whatsoever. Exiting!")
        exit()


if __name__ == "__main__":
    run_spock()
