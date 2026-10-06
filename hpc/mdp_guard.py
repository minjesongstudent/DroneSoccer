"""Decide whether a new src/contract.py changes only the MDP the autopilot may tune.

    python hpc/mdp_guard.py <original contract.py> <new contract.py>

Prints each violation and exits 1, or exits 0 when the change is allowed.
autopilot_hook.py runs the same check on every edit before it is written, and
autopilot_guard.py runs it again on the final file, so nothing depends on the
autopilot Claude following its instructions.

Allowed to differ:
    numeric constants named R_* (edit, add or remove) and GAMMA,
    the body of reward(), whose signature stays fixed,
    OBS_LAGS, which frames of the history the policy sees (see _check_lags).
Everything else in contract.py must parse to the same AST: the observation
code the real detector shares, CRASHES, LAPS_MAX_STEPS, imports, and so on.
Comments and blank lines are free, since the AST ignores them.

reward() is sandboxed by a whitelist, not a blacklist: it may read its own
arguments, its own locals, R_* / GAMMA / CRASHES, a few builtins and a few
pure numpy functions. That rules out imports, file or network access, and
reaching the sim through an attribute.
"""
from __future__ import annotations

import ast
import math
import re
import sys

TUNABLE = re.compile(r"^(R_[A-Z0-9_]+|GAMMA|OBS_LAGS)$")
# OBS_LAGS: frames back in time, oldest first, newest (0) last. baseline.Lapper
# reads lags 0, 1, 2, 4 and 8, so those stay. The frame deque and the policy
# input grow with the largest lag and the count, so both are capped.
LAGS_REQUIRED = {0, 1, 2, 4, 8}
LAGS_MAX = 100        # 10 s at 10 Hz
LAGS_COUNT_MAX = 16
FUNCTION = "reward"
READABLE_CONSTANTS = re.compile(r"^(R_[A-Z0-9_]+|GAMMA|CRASHES)$")
BUILTINS = {"abs", "min", "max", "float", "int", "bool"}
NUMPY = {"clip", "tanh", "exp", "log", "abs", "sqrt", "minimum", "maximum", "square", "sign",
         "hypot", "pi"}
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Lambda,
                   ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.With, ast.AsyncWith,
                   ast.Await, ast.Yield, ast.YieldFrom, ast.Delete, ast.Try)


def _assigned_name(node) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _split(tree: ast.Module):
    """The frozen top-level statements, the tunable constants, and reward()."""
    frozen, constants, function = [], {}, None
    for node in tree.body:
        name = _assigned_name(node)
        if name and TUNABLE.match(name):
            constants[name] = node
        elif isinstance(node, ast.FunctionDef) and node.name == FUNCTION:
            function = node
        else:
            frozen.append(node)
    return frozen, constants, function


def _number(node) -> float | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _number(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub) else inner)
    if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):
        return float(node.value)
    return None


def _check_lags(node) -> list[str]:
    """OBS_LAGS must be a tuple literal of ints, strictly decreasing, ending in 0."""
    if not isinstance(node, ast.Tuple):
        return ["OBS_LAGS must be a tuple literal of ints"]
    lags = []
    for element in node.elts:
        if not (isinstance(element, ast.Constant) and type(element.value) is int):
            return ["OBS_LAGS must be a tuple literal of ints"]
        lags.append(element.value)
    errors = []
    if not lags or lags[-1] != 0 or any(a <= b for a, b in zip(lags, lags[1:])):
        errors.append("OBS_LAGS must be strictly decreasing (oldest first) and end in 0 (now)")
    if lags and max(lags) > LAGS_MAX:
        errors.append(f"OBS_LAGS may reach back at most {LAGS_MAX} steps")
    if len(lags) > LAGS_COUNT_MAX:
        errors.append(f"OBS_LAGS may hold at most {LAGS_COUNT_MAX} frames")
    missing = LAGS_REQUIRED - set(lags)
    if missing:
        errors.append(f"OBS_LAGS must keep lags {sorted(LAGS_REQUIRED)}, which baseline.py reads; "
                      f"missing {sorted(missing)}")
    return errors


def _check_function(new: ast.FunctionDef, old: ast.FunctionDef) -> list[str]:
    errors = []
    if ast.dump(new.args) != ast.dump(old.args) or ast.dump(new.returns or ast.Pass()) != \
            ast.dump(old.returns or ast.Pass()):
        errors.append("reward(): the signature must not change")
    if new.decorator_list:
        errors.append("reward(): no decorators")
    arguments = {a.arg for a in new.args.args + new.args.kwonlyargs}
    # The signature (and its np.ndarray annotations) is checked above; police the body.
    body = [node for stmt in new.body for node in ast.walk(stmt)]
    stored = {n.id for n in body if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for node in body:
        if isinstance(node, FORBIDDEN_NODES):
            errors.append(f"reward() line {node.lineno}: {type(node).__name__} is not allowed")
        elif isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr in NUMPY):
                errors.append(f"reward() line {node.lineno}: attribute .{node.attr} is not allowed; "
                              f"only np.{{{', '.join(sorted(NUMPY))}}}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            ok = (node.id in arguments or node.id in stored or node.id in BUILTINS
                  or node.id == "np" or READABLE_CONSTANTS.match(node.id))
            if not ok or node.id.startswith("__"):
                errors.append(f"reward() line {node.lineno}: name {node.id!r} is not allowed")
        elif isinstance(node, ast.Call):
            f = node.func
            if not ((isinstance(f, ast.Name) and f.id in BUILTINS) or isinstance(f, ast.Attribute)):
                errors.append(f"reward() line {node.lineno}: only builtins "
                              f"{sorted(BUILTINS)} and np functions may be called")
    return errors


def check(original: str, new: str) -> list[str]:
    """Every reason `new` is not an MDP-only change of `original`; empty if allowed."""
    try:
        new_tree = ast.parse(new)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    old_frozen, _, old_function = _split(ast.parse(original))
    new_frozen, constants, new_function = _split(new_tree)
    errors = []
    if ast.dump(ast.Module(old_frozen, [])) != ast.dump(ast.Module(new_frozen, [])):
        errors.append("contract.py changed outside the MDP surface (only R_* constants, GAMMA "
                      "and the body of reward() may change): " + _first_difference(old_frozen, new_frozen))
    for name, node in constants.items():
        if name == "OBS_LAGS":
            errors += _check_lags(node.value)
            continue
        value = _number(node.value)
        if value is None or not math.isfinite(value):
            errors.append(f"{name} must be a finite number literal")
        elif name == "GAMMA" and not 0.0 < value < 1.0:
            errors.append("GAMMA must be between 0 and 1")
    if "OBS_LAGS" not in constants:
        errors.append("OBS_LAGS must stay defined")
    if "GAMMA" not in constants:
        errors.append("GAMMA must stay defined")
    if new_function is None:
        errors.append("reward() must stay defined")
    elif old_function is not None:
        errors += _check_function(new_function, old_function)
    return errors


def _first_difference(old: list, new: list) -> str:
    for a, b in zip(old, new):
        if ast.dump(a) != ast.dump(b):
            return f"first at line {getattr(b, 'lineno', '?')} of the new file"
    return f"{abs(len(old) - len(new))} top-level statement(s) added or removed"


if __name__ == "__main__":
    original, new = (open(p).read() for p in sys.argv[1:3])
    problems = check(original, new)
    for p in problems:
        print(p)
    sys.exit(1 if problems else 0)
