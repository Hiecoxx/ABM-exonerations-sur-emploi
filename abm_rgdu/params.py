"""Parameters, read once from the versioned JSON file.

A run never goes back to the file. Each family of parameters carries a status
in the registry: R for an official rule, E for an estimate checked against a
source, H for a hypothesis that has to be tested rather than believed.
"""

import copy
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "params_v1_5.json"


def load(path=DEFAULT_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def override(params, changes):
    """Copy of `params` with dotted keys replaced, e.g. {"demand.eta": 0.0}.

    Unknown keys raise, so that a typo in a sensitivity grid cannot pass
    silently for a result.
    """
    out = copy.deepcopy(params)
    for dotted, value in changes.items():
        node = out
        *path, last = dotted.split(".")
        for key in path:
            node = node[key]
        if last not in node:
            raise KeyError(f"unknown parameter: {dotted}")
        node[last] = value
    return out


def status(params):
    """The label every output carries."""
    return params["meta"]["status"]


def registry_rows(params):
    return [{"family": family, **entry} for family, entry in params["registry"].items()]
