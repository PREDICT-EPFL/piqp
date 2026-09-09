# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["clarabel==0.11.1", "numpy==2.2.6", "scipy==1.15.3"]
# ///
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import clarabel
import numpy as np
from scipy import sparse


def read(path):
    tokens = iter(path.read_text().split())

    def integer():
        return int(next(tokens))

    def matrix():
        rows, cols = integer(), integer()
        return np.array([float(next(tokens)) for _ in range(rows * cols)]).reshape(rows, cols)

    P, c, A, b = matrix(), matrix().ravel(), matrix(), matrix().ravel()
    n = len(c)
    quadratic, cones = [], []
    for _ in range(integer()):
        ids = [integer() for _ in range(integer())] or list(range(n))
        Q, q, upper = matrix(), matrix().ravel(), float(next(tokens))
        quadratic.append((ids, Q, q, upper))
    for _ in range(integer()):
        rotated = integer()
        ids = [integer() for _ in range(integer())] or list(range(n))
        cones.append((ids, matrix(), matrix().ravel(), rotated))
    return P, c, A, b, quadratic, cones


def read_public(path):
    tokens = iter(path.read_text().split())

    def integer():
        return int(next(tokens))

    def vector(size):
        return np.array([float(next(tokens)) for _ in range(size)])

    def matrix():
        rows, cols, count = integer(), integer(), integer()
        indices = [(integer(), integer(), float(next(tokens))) for _ in range(count)]
        if not indices:
            return sparse.csc_matrix((rows, cols))
        i, j, values = zip(*indices)
        return sparse.csc_matrix((values, (i, j)), shape=(rows, cols))

    n = integer()
    c = vector(n)
    A = matrix()
    b = vector(A.shape[0])
    G = matrix()
    h = vector(G.shape[0])
    cones = []
    for _ in range(integer()):
        rotated, support = integer(), integer()
        ids = [integer() for _ in range(support)]
        F = matrix()
        cones.append((ids, F, vector(F.shape[0]), rotated))
    for i in range(len(h)):
        row = G.getrow(i)
        cones.append((row.indices.tolist(), np.vstack([-row.data, np.zeros(row.nnz)]), np.array([h[i], 0.]), False))
    return sparse.csc_matrix((n, n)), c, A, b, [], cones


def rotate(F, f):
    out_f = np.sqrt(2) * f.copy()
    if sparse.issparse(F):
        out_F = sparse.vstack([F.getrow(0) + F.getrow(1), F.getrow(0) - F.getrow(1), np.sqrt(2) * F[2:]]).tocsc()
    else:
        out_F = np.sqrt(2) * F.copy()
        out_F[0], out_F[1] = F[0] + F[1], F[0] - F[1]
    out_f[0], out_f[1] = f[0] + f[1], f[0] - f[1]
    return out_F, out_f


def convert(data):
    P, c, A, b, quadratic, cones = data
    n = len(c)
    matrices, offsets, kinds = [sparse.csc_matrix(A)], [b], [clarabel.ZeroConeT(len(b))]
    for ids, Q, q, upper in quadratic:
        values, vectors = np.linalg.eigh(Q)
        assert values.min() >= -1e-12
        L = np.sqrt(np.maximum(values, 0))[:, None] * vectors.T
        F = np.vstack([-q, np.zeros(len(q)), L])
        f = np.r_[upper, 1.0, np.zeros(len(q))]
        cones = cones + [(ids, F, f, True)]
    for ids, F, f, rotated in cones:
        if rotated:
            F, f = rotate(F, f)
        coordinates = sparse.coo_matrix(F)
        full = sparse.csc_matrix((coordinates.data, (coordinates.row, np.asarray(ids)[coordinates.col])), shape=(len(f), n))
        matrices.append(-full)
        offsets.append(f)
        kinds.append(clarabel.SecondOrderConeT(len(f)))
    return sparse.triu(sparse.csc_matrix(P)).tocsc(), c, sparse.vstack(matrices).tocsc(), np.concatenate(offsets), kinds


def product(matrix, vector):
    return matrix @ vector if sparse.issparse(matrix) else np.einsum("ij,j->i", matrix, vector, optimize=False)


def violation(data, x):
    P, c, A, b, quadratic, cones = data
    residual = float(np.max(np.abs(product(A, x) - b), initial=0))
    for ids, Q, q, upper in quadratic:
        u = x[ids]
        residual = max(residual, .5 * np.sum(u * product(Q, u)) + np.sum(q * u) - upper)
    for ids, F, f, rotated in cones:
        if rotated:
            F, f = rotate(F, f)
        s = product(F, x[ids]) + f
        residual = max(residual, np.linalg.norm(s[1:]) - s[0])
    return residual


def dual_metrics(data, converted, x, z, slack):
    P, c, A, b, quadratic, cones = data
    rd = product(P, x) + c + product(A.T, z[:len(b)])
    cone_violation = 0.0
    offset = len(b)
    for ids, F, f, rotated in cones:
        dual = z[offset:offset + len(f)].copy()
        if rotated:
            a, second = dual[:2]
            dual *= np.sqrt(2)
            dual[:2] = [a + second, a - second]
        rd[ids] -= product(F.T, dual)
        offset += len(f)
    for ids, Q, q, upper in quadratic:
        multiplier = z[offset] + z[offset + 1]
        rd[ids] += multiplier * (product(Q, x[ids]) + q)
        cone_violation = max(cone_violation, -multiplier)
        offset += len(q) + 2
    assert offset == len(z)
    offset = 0
    for cone in converted[4]:
        block = z[offset:offset + cone.dim]
        if isinstance(cone, clarabel.SecondOrderConeT):
            cone_violation = max(cone_violation, np.linalg.norm(block[1:]) - block[0])
        elif isinstance(cone, clarabel.NonnegativeConeT):
            cone_violation = max(cone_violation, -np.min(block, initial=0))
        offset += cone.dim
    slack_error = float(np.max(np.abs(converted[2] @ x + slack - converted[3]), initial=0))
    conic_rd = product(P, x) + c + converted[2].T @ z
    return (float(np.max(np.abs(conic_rd), initial=0)), float(np.max(np.abs(rd), initial=0)),
            cone_violation, slack_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", type=Path)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--piqp-csv", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-11)
    parser.add_argument("--filter", default="")
    args = parser.parse_args()
    reference = {}
    if args.piqp_csv:
        for row in csv.DictReader(args.piqp_csv.open()):
            if row["valid"] == "1":
                reference[row["instance"]] = float(row["objective"])
    if args.metadata:
        args.metadata.write_text(json.dumps({"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version, "clarabel": clarabel.__version__, "numpy": np.__version__,
            "requested_tolerance": args.tolerance, "repetitions": args.repetitions, "command": sys.argv,
            "threads": {k: os.environ[k] for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]}}, indent=2) + "\n")
    writer = csv.writer(sys.stdout)
    writer.writerow(["instance", "backend", "repetition", "n", "status", "iterations", "setup_s", "solve_s",
                     "primal_violation", "dual_residual", "native_dual_residual", "native_valid", "dual_cone_violation", "slack_error", "complementarity", "normalized_complementarity", "gap", "objective", "objective_error", "valid"])
    failed = False
    for path in sorted(list(args.instances.glob("*.dat")) + list(args.instances.glob("*.socp"))):
        if args.filter not in path.stem:
            continue
        data = read_public(path) if path.suffix == ".socp" else read(path)
        converted = convert(data)
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 150
        settings.max_threads = 1
        settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = args.tolerance
        start = time.perf_counter()
        solver = clarabel.DefaultSolver(*converted, settings)
        setup = time.perf_counter() - start
        for repetition in range(-1, args.repetitions):
            start = time.perf_counter()
            solution = solver.solve()
            elapsed = time.perf_counter() - start
            if repetition < 0:
                continue
            info = solver.get_info()
            x = np.array(solution.x)
            feasibility = violation(data, x)
            objective = .5 * np.sum(x * product(data[0], x)) + np.sum(data[1] * x)
            error = abs(objective - reference[path.stem]) if path.stem in reference else float("nan")
            z, slack = np.asarray(solution.z), np.asarray(solution.s)
            dual, native_dual, cone_violation, slack_error = dual_metrics(data, converted, x, z, slack)
            feasibility = max(feasibility, slack_error)
            complementarity = float(np.sum(z * slack))
            normalized_complementarity = complementarity / (1 + abs(objective))
            roundoff = 64 * np.finfo(float).eps * (1 + np.sum(np.abs(z * slack)))
            valid = (str(solution.status) in ("Solved", "AlmostSolved") and np.isfinite(objective)
                     and np.all(np.isfinite(x)) and np.all(np.isfinite(z)) and np.all(np.isfinite(slack))
                     and feasibility <= 1e-7 and dual <= 1e-7 and cone_violation <= 1e-7
                     and -roundoff <= complementarity and normalized_complementarity <= 1e-7)
            native_valid = valid and native_dual <= 1e-7
            failed |= not valid
            writer.writerow([path.stem, "clarabel", repetition, len(x), solution.status, solution.iterations,
                             setup, elapsed, feasibility, dual, native_dual, int(native_valid), cone_violation, slack_error, complementarity,
                             normalized_complementarity, info.gap_abs, objective, error, int(valid)])
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
