#!/usr/bin/env python

import argparse
import itertools
import os
import sys
import re
import sklearn as sk
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


def n_iter_helper(ok):
    if ok:
        return 200
    if not ok:
        return 1000


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
    return re.sub("[^a-zA-Z0-9 \n\.]", "_", filename).replace(" ", "_")


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def reweighter(target, wp=2):
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


def processargs(arguments):
    vbuilder = argparse.ArgumentParser(
        prog="spock",
        description="Fit volcano plots to experimental data.",
        epilog="Remember to cite the spock paper (when its out!) \n \n - and enjoy!",
    )
    vbuilder.add_argument(
        "-version", "--version", action="version", version="%(prog)s 0.0.1"
    )
    runmode_arg = vbuilder.add_mutually_exclusive_group()
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
        default=2,
        help="In the regression, integer power with which higher activity points are weighted. Higher means low activity points are given less priority in the fit (default: 2)",
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
        "-is",
        "--is",
        dest="imputer_strat",
        type=str,
        default="none",
        help="Imputter to refill missing datapoints. Beta version. (default: None)",
    )
    args = vbuilder.parse_args(arguments)

    dfs = check_input(args.filenames, args.wp, args.imputer_strat, args.verb)
    if len(dfs) == 0:
        raise InputError("No input data detected. Exiting.")
    else:
        df = dfs[0]
    assert isinstance(df, pd.DataFrame)
    return (df, args.wp, args.verb, args.imputer_strat, args.plotmode)


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
    if not wp > 0 and isinstance(wp, int):
        raise InputError("Invalid weighting power input! Should be a positive integer.")
    return dfs
