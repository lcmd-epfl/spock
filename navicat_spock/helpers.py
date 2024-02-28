#!/usr/bin/env python

import argparse
import itertools
import os
from itertools import cycle

import numpy as np
import pandas as pd

from navicat_volcanic.exceptions import InputError


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
        prog="volcanic",
        description="Build volcano plots from reaction energy profile data.",
        epilog="Remember to cite the volcanic paper: \n \nLaplaza, R., Das, S., Wodrich, M.D. et al. Constructing and interpreting volcano plots and activity maps to navigate homogeneous catalyst landscapes. Nat Protoc (2022). \nhttps://doi.org/10.1038/s41596-022-00726-2 \n \n - and enjoy!",
    )
    vbuilder.add_argument(
        "-version", "--version", action="version", version="%(prog)s 1.3.3"
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
        help="Plot mode for volcano and activity map plotting. Higher is more detailed, lower is basic. 3 includes uncertainties. (default: 1)",
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

    dfs, ddfs = check_input(
        args.filenames,
        args.imputer_strat,
        args.verb,
    )
    if len(dfs) == 0:
        raise InputError("No input profiles detected in file. Exiting.")
    else:
        df = dfs[0]
    assert isinstance(df, pd.DataFrame)
    if args.verb > 1:
        print("Final descriptor database (top rows):")
        print(ddf.head())
    for column in ddf:
        df.insert(1, f"Descriptor {column}", ddf[column].values)
    return (
        df,
        args.verb,
        args.temp,
        args.imputer_strat,
        args.plotmode,
    )


def check_input(filenames, imputer_strat, verb):
    accepted_excel_terms = ["xls", "xlsx"]
    accepted_imputer_strats = ["simple", "knn", "iterative", "none"]
    accepted_nds = [1, 2]
    dfs = []
    ddfs = []
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
    return dfs, ddfs
