#!/usr/bin/env python

import argparse
import itertools
import os
import sys
import re
import sklearn
from sklearn import linear_model
from itertools import cycle
from io import StringIO

import numpy as np
import pandas as pd

from navicat_spock.exceptions import InputError


def call_imputer(a, b, imputer_strat="iterative"):
    if imputer_strat == "iterative":
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
        except ModuleNotFoundError as err:
            return a
        imputer = IterativeImputer(max_iter=25)
        newa = imputer.fit(b).transform(a.reshape(1, -1)).flatten()
        return newa
    elif imputer_strat == "simple":
        try:
            from sklearn.impute import SimpleImputer
        except ModuleNotFoundError as err:
            return a
        imputer = SimpleImputer()
        newa = imputer.fit_transform(a.reshape(-1, 1)).flatten()
        return newa
    elif imputer_strat == "knn":
        try:
            from sklearn.impute import KNNImputer
        except ModuleNotFoundError as err:
            return a
        imputer = KNNImputer(n_neighbors=2)
        newa = imputer.fit(b).transform(a.reshape(1, -1)).flatten()
        return newa
    elif imputer_strat == "none":
        return a
    else:
        return a


def slope_check(alphas, verb=0):
    if verb > 4:
        print(f"Slopes are: {alphas}")
    if alphas is None:
        return False
    if len(alphas) == 1:
        return True
    if len(alphas) == 2:
        if alphas[0] > 0:
            return alphas[1] < 0
        if alphas[0] < 0:
            return alphas[1] > 0
    if len(alphas) == 3:
        if alphas[0] > 0:
            if alphas[1] > 0:
                return alphas[2] < 0
            if alphas[1] < 0:
                return alphas[2] < 0
        if alphas[0] < 0:
            if alphas[1] > 0:
                return alphas[2] > 0
            if alphas[1] < 0:
                return alphas[2] > 0
    return False


def find_duplicated_columns(df):
    dupes = []
    df = df.round(6)
    columns = df.columns
    for i in range(len(columns)):
        col1 = df.iloc[:, i]
        for j in range(i + 1, len(columns)):
            col2 = df.iloc[:, j]
            # break early if dtypes aren't the same (helps deal with
            # categorical dtypes)
            if col1.dtype is not col2.dtype:
                break
            # otherwise compare values
            if col1.equals(col2):
                dupes.append(columns[i])
                break
            # remove constants too
            if col1.nunique() <= 1:
                dupes.append(columns[i])
                break
    return dupes


def fast_vif(X):
    cc = np.corrcoef(X, rowvar=False)
    vif = np.linalg.inv(cc)
    return vif.diagonal()


def prune_by_vif(X, thresh=4, verb=0):
    if verb > 2:
        print(
            f"Augmented features will be pruned based on a variance inflation factor or {thresh}"
        )
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped = True
    np.save("test_1.npy", X[cols].values)
    while dropped:
        dropped = False
        c = X[cols[variables]].values
        vif = np.abs(fast_vif(c))
        maxloc = np.argmax(vif)
        if vif[maxloc] > thresh:
            if verb > 2:
                print(
                    f"Dropping {X[cols[variables]].columns[maxloc]} at index {maxloc} due to max. variance inflation of {vif[maxloc]}."
                )
            variables = np.delete(variables, maxloc)
            dropped = True
        else:
            if verb > 2:
                print(
                    f"Exiting variance inflation pruning due to max. variance inflation of {vif[maxloc]} < {thresh}."
                )
    np.save("test_2.npy", c)
    return X[cols[variables]]


def augment(df, level, verb=0):
    y = df.iloc[:, 0:2]
    x_full = df.iloc[:, 2:]
    if verb > 6:
        print(y.head())
        print(x_full.head())
    if level > 0:
        if verb > 0:
            print(f"Doing level 1 feature augmentation...")
        for i in x_full.keys():
            inv = f"1/{i}"
            pow2 = f"{i}^2"
            # pow3 = f"{i}^3"
            sqrt = f"sqrt({i})"
            # log = f"log({i})"
            x_full[inv] = 1 / x_full[i]
            x_full[pow2] = x_full[i] ** 2
            # x_full[pow3] = x_full[i] ** 3
            if not (x_full[i].values < 0).any():
                x_full[sqrt] = np.sqrt(x_full[i])
                # x_full[log] = np.log(x_full[i])
            if not (abs(x_full[i].values) < 0.01).any():
                x_full[inv] = 1 / x_full[i]
    if level == 2:
        if verb > 0:
            print(f"Doing level 2 feature augmentation...")
        for i, j in zip(x_full.keys(), x_full.keys()[1:]):
            prod = f"{i}x{j}"
            div12 = f"{i}/{j}"
            div21 = f"{j}/{i}"
            x_full[prod] = x_full[i] * x_full[j]
            if "1/" not in i and "1/" not in j:
                x_full[div12] = x_full[i] / x_full[j]
                x_full[div21] = x_full[j] / x_full[i]
    if level == 3:
        if verb > 0:
            print(f"Doing level 3 feature augmentation...")
        for i, j in itertools.combinations_with_replacement(
            range(len(x_full.keys())), 2
        ):
            if i == j:
                continue
            i = x_full.keys()[i]
            j = x_full.keys()[j]
            prod = f"{i}x{j}"
            div12 = f"{i}/{j}"
            div21 = f"{j}/{i}"
            x_full[prod] = x_full[i] * x_full[j]
            if "1/" not in i and "1/" not in j:
                x_full[div12] = x_full[i] / x_full[j]
                x_full[div21] = x_full[j] / x_full[i]
    # Remove almost exact duplicates from feature datafragme, also removes constant columns
    dups = find_duplicated_columns(x_full)
    x_full = x_full.drop(dups, axis=1)
    # Prune redundant features based on vif, we use a very high threshold to be conservative
    # x_full = prune_by_vif(x_full, thresh=10e6, verb=verb)
    if verb > 6:
        print(y.head())
        print(x_full.head())
    df = pd.concat([y, x_full], axis=1)
    if verb > 6:
        print(df.head())
    return clean_df(df)


def clean_df(df):
    assert isinstance(df, pd.DataFrame)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True, how="any", axis=1)
    return df


def n_iter_helper(ok):
    if ok:
        return 1000
    if not ok:
        return 5000


def Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def namefixer(filename):
    assert isinstance(filename, str)
    return re.sub("[^a-zA-Z0-9 \n\.]", "_", filename).replace(" ", "_")


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def reweighter(target, wp=2):
    assert isinstance(wp, int)
    if wp > 0:
        std = target.std()
        norm = sum(target)  # Not needed since robust regression will normalize
        rescaled = [(py - min(target)) + std for py in target]
        # print(rescaled)
        scaled = [(py / max(abs(target))) for py in rescaled]
        # print(scaled)
        weights = np.round(
            np.array([py**wp for py in scaled]), decimals=6
        )  # **2 at least, could be increased
        weights = normalize(weights).reshape(-1)
    else:
        std = target.std()
        norm = sum(target)  # Not needed since robust regression will normalize
        weights = np.array([1 / abs(norm) for _ in target])
    return weights


def curate_d(d, descriptors, cb, ms, names, imputer_strat="none", verb=0):
    assert isinstance(d, np.ndarray)
    curated_d = np.zeros_like(d)
    for i in range(d.shape[0]):
        n_nans = np.count_nonzero(np.isnan(d[i, :]))
        if n_nans > 0:
            tofix = d[i, :]
            if verb > 1:
                print(f"Using the imputer strategy, converted\n {tofix}.")
            toref = d[np.arange(d.shape[0]) != i, :]
            d[i, :] = call_imputer(tofix, toref, imputer_strat)
            if verb > 1:
                print(f"to\n {d[i,:]}.")
        curated_d[i, :] = d[i, :]
    incomplete = np.ones_like(curated_d[:, 0], dtype=bool)
    for i in range(curated_d.shape[0]):
        n_nans = np.count_nonzero(np.isnan(d[i, :]))
        if n_nans > 0:
            if verb > 1:
                print(
                    f"Some of your rows contain {n_nans} undefined values and will not be considered:\n {curated_d[i,:]}"
                )
            incomplete[i] = False
    curated_cb = cb[incomplete]
    curated_ms = ms[incomplete]
    curated_names = names[incomplete]
    curated_d = d[incomplete, :]
    return curated_d, curated_cb, curated_ms, curated_names


def yesno(question):
    """Simple Yes/No Function."""
    prompt = f"{question} ? (y/n): "
    ans = input(prompt).strip().lower()
    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return yesno(question)
    if ans == "y":
        return True
    return False


def bround(x, base: float = 10, type=None) -> float:
    if type == "max":
        return base * np.ceil(x / base)
    elif type == "min":
        return base * np.floor(x / base)
    else:
        tick = base * np.round(x / base)
        return tick


def group_data_points(bc, ec, names):
    try:
        groups = np.array([str(i)[bc:ec].upper() for i in names], dtype=object)
    except Exception as m:
        raise InputError(
            f"Grouping by name characters did not work. Error message was:\n {m}"
        )
    type_tags = np.unique(groups)
    cycol = cycle("bgrcmky")
    cymar = cycle("^ospXDvH")
    cdict = dict(zip(type_tags, cycol))
    mdict = dict(zip(type_tags, cymar))
    cb = np.array([cdict[i] for i in groups])
    ms = np.array([mdict[i] for i in groups])
    return cb, ms


def constant_data_points(names):
    cycol = cycle("b")
    cymar = cycle("o")
    cdict = dict(zip(names, cycol))
    mdict = dict(zip(names, cymar))
    cb = np.array([cdict[i] for i in names])
    ms = np.array([mdict[i] for i in names])
    return cb, ms


def processargs(arguments):
    vbuilder = argparse.ArgumentParser(
        prog="spock",
        description="Fit volcano plots to experimental data.",
        epilog="Remember to cite the spock paper (when its out!) \n \n - and enjoy!",
    )
    vbuilder.add_argument(
        "-version", "--version", action="version", version="%(prog)s 0.0.2"
    )
    vbuilder.add_argument(
        "-i",
        "--i",
        "-input",
        dest="filenames",
        nargs="?",
        action="append",
        type=str,
        required=True,
        help="Filename containing catalyst data. Target metric (y-axis) should be labeled as TARGET in column name. See documentation for input and file formatting questions.",
    )
    vbuilder.add_argument(
        "-wp",
        "--wp",
        "-weights",
        "--weights",
        dest="wp",
        type=int,
        default=1,
        help="In the regression, integer power with which higher activity points are weighted. Higher means low activity points are given less priority in the fit (default: 1)",
    )
    vbuilder.add_argument(
        "-v",
        "--v",
        "--verb",
        dest="verb",
        type=int,
        default=0,
        help="Verbosity level of the code. Higher is more verbose and viceversa. Set to at least 2 to generate csv output files (default: 1)",
    )
    vbuilder.add_argument(
        "-pm",
        "--pm",
        "-plotmode",
        "--plotmode",
        dest="plotmode",
        type=int,
        default=1,
        help="Plot mode for volcano plotting. Higher is more detailed, lower is more basic. (default: 1)",
    )
    vbuilder.add_argument(
        "-rng",
        "--rng",
        "-random",
        "--random",
        dest="seed",
        type=int,
        default=42,
        help="Random seed to use in the Muggeo fits. (default: 42)",
    )
    vbuilder.add_argument(
        "-fa",
        "--fa",
        "-augment",
        "--augment",
        dest="fa",
        type=int,
        default=0,
        help="Level of feature augmentation to perform. Higher is more feature augmentation, 0 is no feature augmentation. (default: 0, no augmentation)",
    )
    vbuilder.add_argument(
        "-is",
        "--is",
        dest="imputer_strat",
        type=str,
        default="none",
        help="Imputter to refill missing datapoints. Beta version. (default: None)",
    )
    vbuilder.add_argument(
        "--plot_all",
        "-plot_all",
        dest="prefit",
        type=bool,
        default=0,
        help="Plot and print the best volcano per descriptor. This is slow and writes a lot of plots, do not use unless you know what you are doing. (default: 0, False)",
    )
    vbuilder.add_argument(
        "--save_fig",
        "-save_fig",
        dest="save_fig",
        type=bool,
        default=0,
        help="Save the volcano plot as an image. (default: 0, False)",
    )
    vbuilder.add_argument(
        "--save_csv",
        "-save_csv",
        dest="save_csv",
        type=bool,
        default=0,
        help="Save the volcano plot data to a CSV file. (default: 0, False)",
    )
    args = vbuilder.parse_args(arguments)

    dfs = check_input(args.filenames, args.wp, args.imputer_strat, args.verb)
    if len(dfs) == 0:
        raise InputError("No input data detected. Exiting.")
    else:
        df = dfs[0]
    assert isinstance(df, pd.DataFrame)
    if args.fa > 0:
        df = augment(df, args.fa, args.verb)
    return (
        df,
        args.wp,
        args.verb,
        args.imputer_strat,
        args.plotmode,
        args.seed,
        args.prefit,
        args.save_fig,
        args.save_csv,
    )


def check_input(filenames, wp, imputer_strat, verb):
    accepted_excel_terms = ["xls", "xlsx"]
    accepted_imputer_strats = ["simple", "knn", "iterative", "none"]
    accepted_nds = [1, 2]
    dfs = []
    for filename in filenames:
        if filename.split(".")[-1] in accepted_excel_terms:
            dfs.append(pd.read_excel(filename))
        elif filename.split(".")[-1] == "csv":
            dfs.append(pd.read_csv(filename))
        else:
            raise InputError(
                f"File termination for filename {filename} was not understood. Try csv or one of {accepted_excel_terms}."
            )
    if imputer_strat not in accepted_imputer_strats:
        raise InputError(
            f"Invalid imputer strat in input!\n Accepted values are:\n {accepted_imputer_strats}"
        )
    if not isinstance(verb, int):
        raise InputError("Invalid verbosity input! Should be a positive integer or 0.")
    if not isinstance(wp, int):
        raise InputError(
            "Invalid weighting power input! Should be a positive integer or 0."
        )
    elif wp < 0:
        raise InputError(
            "Invalid weighting power input! Should be a positive integer or 0."
        )
    return dfs


def test_vif():
    sol = np.array([22.95, 3.0, 12.95, 3.0])
    a = [1, 1, 2, 3, 4]
    b = [2, 2, 3, 2, 1]
    c = [4, 6, 7, 8, 9]
    d = [4, 3, 4, 5, 4]
    ck = np.column_stack([a, b, c, d])
    # print(ck)
    cc = np.corrcoef(ck, rowvar=False)
    vifm = np.linalg.inv(cc)
    vif = vifm.diagonal()
    assert np.allclose(vif, sol)


def test_slope_check():
    assert not slope_check([10, 10, 10])
    assert not slope_check([10, -10, 10])
    assert not slope_check([-10, 10, -10])
    assert slope_check([10, 10, -10])
    assert slope_check([10, -10, -10])


if __name__ == "__main__":
    test_vif()
