#!/usr/bin/env python

import sys

from navicat_spock.helpers import (
    processargs,
    group_data_points,
    setflags,
)
from navicat_spock.piecewise_regression import (
    ModelSelection,
    Fit,
)


def run_spock():
    (
        df,
        verb,
        imputer_strat,
        plotmode,
    ) = processargs(sys.argv[1:])

    # Fill in reaction profile names/IDs from input data.
    if verb > 0:
        print(
            f"volcanic will assume that {df.columns[0]} contains names/IDs of reaction profiles."
        )
    names = df[df.columns[0]].values

    # Atttempts to group data points based on shared characters in names.
    cb, ms = group_data_points(ic, fc, names)

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
                print(f"Assuming field {tag} corresponds to a non-TS stationary point.")
            descriptors[i] = True

    # Your data might contain outliers (human error, computation error) or missing points.
    # We will attempt to curate your data automatically.
    # d, cb, ms = curate_d(d, descriptor, cb, ms, tags, imputer_strat, nstds=3, verb=verb)

    descriptor = d[:, idx]  # .reshape(-1)
    target = d[:, tidx]  # .reshape(-1)

    # Weights
    norm = sum(target)
    weights = np.array([abs(py) ** 3 for py in target])
    for y, w in zip(target, weights):
        # print(y,w)
        continue
    ms = ModelSelection(
        data["x"].values,
        data["y"].values,
        max_breakpoints=2,
        max_iterations=100,
        weights=weights,
    )
    bic_list = np.array([summary["bic"] for summary in ms.model_summaries], dtype=float)
    n_list = np.array(
        [summary["n_breakpoints"] for summary in ms.model_summaries], dtype=int
    )
    filter_nan = ~np.isnan(bic_list)
    bic_list = bic_list[filter_nan]
    n_list = n_list[filter_nan]
    n = int(n_list[np.argmin(bic_list)])
    print(f"The best number of breakpoints according to BIC is {n}")

    # Fit piecewise regression!
    pw_fit = Fit(descriptor, target, n_breakpoints=n)
    pw_fit.summary()
    # Plot the data, fit, breakpoints and confidence intervals
    pw_fit.plot_data(color="grey", s=20)
    # Pass in standard matplotlib keywords to control any of the plots
    pw_fit.plot_fit(color="red", linewidth=4)
    pw_fit.plot_breakpoints()
    pw_fit.plot_breakpoint_confidence_intervals()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    plt.close()
