#!/usr/bin/env python

import sys
import traceback

import numpy as np
import matplotlib.pyplot as plt

from navicat_spock.helpers import (
    processargs,
    group_data_points,
    constant_data_points,
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
    (
        df,
        wp,
        verb,
        imputer_strat,
        plotmode,
        seed,
        prefit,
        save_fig,
        save_csv,
    ) = processargs(sys.argv[1:])
    _ = run_spock_from_args(
        df=df,
        wp=wp,
        verb=verb,
        imputer_strat=imputer_strat,
        plotmode=plotmode,
        seed=seed,
        prefit=prefit,
        save_fig=save_fig,
        save_csv=save_csv,
    )


def run_spock_from_args(
    df,
    wp=2,
    verb=0,
    imputer_strat="none",
    plotmode=1,
    seed=None,
    prefit=False,
    setcbms=True,
    fig=None,
    ax=None,
    save_fig=True,
    save_csv=True,
):
    if seed is None:
        seed = int(np.random.rand() * (2**32 - 1))
    np.random.seed(0)
    if verb > 0:
        print(f"Random seed set to {seed}.")
        print(
            f"spock will assume that {df.columns[0]} contains names/IDs of catalysts/samples."
        )
    names = df[df.columns[0]].values
    if fig is None and ax is None:
        fig, ax = plt.subplots(
            frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
        )

    # Atttempts to group data points based on shared characters in names.
    if setcbms:
        cb, ms = group_data_points(0, 2, names)
    else:
        cb, ms = constant_data_points(names)

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
    msels = []
    for i, idx in enumerate(idxs):
        fitted = False
        try:
            if verb > 0:
                print(f"Attempting fit with descriptor index {idx}: {tags[idx]}...:")
            descriptor = d[:, idx].reshape(-1)
            xthresh = 0.05 * (max(descriptor) - min(descriptor))
            msel = ModelSelection(
                descriptor,
                target,
                max_breakpoints=max_breakpoints,
                max_iterations=n_iter_helper(fitted),
                weights=weights,
                tolerance=xthresh,
                verbose=verb > 2,
            )
            msels.append(msel)
            all_sc[i, :] = np.array(
                [
                    slope_check(summary["slopes"], verb)
                    for summary in msel.model_summaries
                ],
                dtype=bool,
            )
            all_bic[i, :] = np.array(
                [summary["bic"] for summary in msel.model_summaries], dtype=float
            )
            all_n[i, :] = np.array(
                [summary["n_breakpoints"] for summary in msel.model_summaries],
                dtype=int,
            )

            # Save BICs and ns just in case
            bic_list = all_bic[i, :]
            n_list = all_n[i, :]
            sc_list = all_sc[i, :]

            if verb > 4:
                print(
                    f"The list of BICs for n breakpoints are:\n {bic_list} for\n {n_list}"
                )

            # Find best n
            n = n_list[np.nanargmin(bic_list)]
            sc = sc_list[np.nanargmin(bic_list)]
            min_bic = np.min(bic_list)

            if n < 1:
                if verb > 1:
                    print(
                        f"BIC seems to indicate that a linear fit is better than a volcano fit for this descriptor. Warning!"
                    )
            else:
                fitted = True

            if prefit and fitted and n > 0 and sc:
                if verb > 3:
                    print(
                        f"Prefitting the best model for this descriptor with {n} breakpoints..."
                    )
                # Fit piecewise regression!
                pw_fit = Fit(
                    descriptor,
                    target,
                    n_breakpoints=int(n),
                    weights=weights,
                    max_iterations=n_iter_helper(fitted),
                    tolerance=xthresh,
                )
                if verb > 2:
                    pw_fit.summary()
                if not pw_fit.best_muggeo:
                    if verb > 2:
                        print(
                            "Prefitting did not work. This is likely a bug or extremely bad luck in the Muggeo fit."
                        )
                    raise ConvergenceError("The fitting process did not converge.")
                # Plot the data and fit for this prefitted model, even if its not great
                if verb > 2:
                    print(
                        f"Prefitting volcano with {n} breakpoints and descriptor index {idx}: {tags[idx]}, for which a BIC of {min_bic} was obtained."
                    )
                _, _ = plot_and_save(
                    pw_fit,
                    tags,
                    idx,
                    tidx,
                    cb,
                    ms,
                    plotmode=plotmode,
                    fig=fig,
                    ax=ax,
                    return_value=False,
                )

        except Exception as m:
            traceback.print_exc()
            all_bic[i, :] = np.inf
            all_n[i, :] = 0
            all_sc[i, :] = False
            if verb > 0:
                print(
                    f"Fit did not converge with descriptor index {idx}: {tags[idx]}\n due to {m}"
                )
            msels.append(None)

    # Done iterating over descriptors
    best_n = np.zeros_like(idxs, dtype=int)
    best_bic = np.zeros_like(idxs, dtype=float)
    best_sc = np.zeros_like(idxs, dtype=bool)
    for i, idx in enumerate(idxs):
        if verb > 4:
            print(f"Filtering fits with descriptor index {idx}: {tags[idx]}...:")

        bic_list = all_bic[i, :]
        n_list = all_n[i, :]
        sc_list = all_sc[i, :]

        if verb > 4:
            print(
                f"Before filtering, the list of BICs for n breakpoints is:\n {bic_list}\n {n_list}\n {sc_list}"
            )

        filter_nan = ~np.isnan(bic_list)
        bic_list = bic_list[filter_nan]
        n_list = n_list[filter_nan]
        sc_list = sc_list[filter_nan]

        if any(sc_list):
            if verb > 4:
                print("Filtering by slope criterion...")
            bic_list = bic_list[sc_list]
            n_list = n_list[sc_list]
        # if any(n_list):
        #    if verb > 4:
        #        print("Filtering by number of breakpoints...")
        #    filter_0s = np.nonzero(n_list)
        #    bic_list = bic_list[filter_0s]
        #    n_list = n_list[filter_0s]
        if verb > 4:
            print(
                f"After filtering, the list of BICs for n breakpoints is:\n {bic_list}\n {n_list}"
            )

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
            xthresh = 0.05 * (max(descriptor) - min(descriptor))
            pw_fit = Fit(
                descriptor,
                target,
                n_breakpoints=n,
                weights=weights,
                max_iterations=n_iter_helper(fitted),
                tolerance=xthresh,
            )
            if not pw_fit.best_muggeo:
                # If for some reason the fit fails now, we use the preliminary fit instead
                pw_fit.best_muggeo = msels[idx - 1].models[n - 1].best_muggeo
            if not slope_check(pw_fit.get_results()["slopes"], verb):
                # If for some reason the fit switched the slopes, we use the preliminary fit instead
                pw_fit.best_muggeo = msels[idx - 1].models[n - 1].best_muggeo
            if verb > 2:
                pw_fit.summary()
            # Plot the data, fit, breakpoints and confidence intervals
            fig, ax = plot_and_save(
                pw_fit,
                tags,
                idx,
                tidx,
                cb,
                ms,
                fig=fig,
                ax=ax,
                plotmode=plotmode,
                save_fig=save_fig,
                save_csv=save_csv,
            )
            return fig, ax
        else:
            min_bic = np.min(best_bic)
            idx = idxs[np.argmin(best_bic)]
            n = int(best_n[np.argmin(best_bic)])
            if verb > 3:
                print(
                    f"Considering n=0 solutions, {n} breakpoints for index {idx}: {tags[idx]} should be used. This does not correspond to a volcano. Exiting!"
                )
            sys.exit(1)
    else:
        print("None of the descriptors could be fit whatsoever. Exiting!")
        sys.exit(1)


if __name__ == "__main__":
    run_spock()
