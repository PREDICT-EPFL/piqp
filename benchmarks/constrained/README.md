# Constrained solver experiments

Run from the repository root. The recorded machine is an Apple M4 Max MacBook with 48 GiB memory, macOS 26.6.2, Apple clang 21, Eigen 5.0.1, and BLASFEO at `/opt/blasfeo_arm64`. Solvers use double precision and one thread. Run each timing batch alone, with no concurrent builds or tests. The measurements use normal macOS scheduling, without core pinning, temperature control, frequency locking, or thermal monitoring.

## Build and run

```sh
cmake -S . -B build/constrained-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON -DBUILD_EXAMPLES=OFF -DBUILD_C_INTERFACE=OFF \
  -DBUILD_WITH_BLASFEO=ON -Dblasfeo_DIR=/opt/blasfeo_arm64 \
  -DBUILD_WITH_TEMPLATE_INSTANTIATION=OFF -DBUILD_WITH_TRACY=OFF
CCACHE_DISABLE=1 cmake --build build/constrained-release --target constrained_solver_benchmark constrained_qp_regression constrained_test -j 4
build/constrained-release/tests/constrained_test
mkdir -p benchmarks/constrained/instances benchmarks/constrained/results
for plate_size in 30 60; do
  curl -fL "https://cblib.zib.de/download/all/nql${plate_size}.cbf.gz" -o "benchmarks/constrained/public/nql${plate_size}.cbf.gz"
done
python3 benchmarks/constrained/import_cblib.py benchmarks/constrained/public/nql30.cbf.gz benchmarks/constrained/public/nql60.cbf.gz --output benchmarks/constrained/instances
build/constrained-release/benchmarks/constrained_solver_benchmark 30 benchmarks/constrained/instances > benchmarks/constrained/results/piqp-generated.csv
build/constrained-release/benchmarks/constrained_solver_benchmark 10 benchmarks/constrained/instances public benchmarks/constrained/instances/nql30.socp benchmarks/constrained/instances/nql60.socp > benchmarks/constrained/results/piqp-public.csv
```

The [Conic Benchmark Library](https://cblib.zib.de/) source URLs and SHA256 hashes are in `public/sources.json`. The importer preserves sparse data and rejects integer variables and unsupported Conic Benchmark Format sections. The 18 generated problems cover paired quadratic/conic ellipsoids, local mixed constraints, double-integrator horizons, and a cone-dimension sweep at fixed two-variable support. The two public plastic-plate problems run only through the general sparse backend.

Combine the PIQP rows for the independent reference run:

```sh
python3 - <<'PY'
from pathlib import Path
root = Path('benchmarks/constrained/results')
Path('/private/tmp/piqp-reference-input.csv').write_text(
    (root / 'piqp-generated.csv').read_text() +
    '\n'.join((root / 'piqp-public.csv').read_text().splitlines()[1:]) + '\n')
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-project --python 3.13 --script benchmarks/constrained/compare.py benchmarks/constrained/instances --repetitions 30 --piqp-csv /private/tmp/piqp-reference-input.csv --metadata benchmarks/constrained/results/reference-environment.json > benchmarks/constrained/results/clarabel.csv
```

Each constrained problem/backend receives one warm-up solve, followed by 30 recorded generated solves, 10 public PIQP solves, or 30 Clarabel solves across all 20 problems. Every solve starts from scratch. `setup_s` is one observation per problem/backend. `update_s` reapplies unchanged bounds and cone offsets; `factor_s` includes assembly and factorization. Validation, export, and symbolic structure analysis are outside solve timers. Clarabel timing includes Python call overhead and excludes conic conversion. Sparse entry counts exclude index arrays and workspaces; they are not peak memory measurements. Dense and multistage counts are `-1`.

Acceptance requires primal feasibility, stationarity, dual-cone violation, and slack consistency at most `1e-7`; complementarity divided by `1 + abs(objective)` at most `1e-7`; finite primal/dual variables; and nonnegative complementarity up to rounding error. Generated PIQP solutions also require backend and available analytic agreement within `1e-4`. PIQP requests `1e-8` absolute/relative tolerances; Clarabel requests `1e-11` because its internal stopping criteria can permit larger original complementarity. Actual `Solved`/`AlmostSolved` statuses remain in the CSV. Clarabel `valid` checks its supplied conic formulation; `native_dual_residual` and `native_valid` separately check recovered quadratic multipliers. Retain these columns and absolute objective errors when comparing direct quadratic constraints with cone lifts. Report dispersion and failures alongside medians; repetitions are not independent problem instances.

## Unchanged QP path

The baseline uses archived headers at revision `4163064033e241a99ddfe983df96610b5bb11573`, with the same compiler, Eigen, and BLASFEO as the current build. The maintained driver requests `1e-10` absolute/relative tolerances, checks the analytic solution, and performs ten warm-ups before 500 recorded solves per case in each of three batches. The batch order alternates baseline/current, current/baseline, baseline/current; both CSVs retain a `batch` column.

```sh
mkdir -p /private/tmp/piqp-baseline
git archive 4163064033e241a99ddfe983df96610b5bb11573 include | tar -x -C /private/tmp/piqp-baseline
/usr/bin/c++ -O3 -DNDEBUG -Wall -Wextra -Wconversion -pedantic -DPIQP_HAS_BLASFEO \
  -I/private/tmp/piqp-baseline/include -isystem /opt/homebrew/include/eigen3 \
  -isystem /opt/blasfeo_arm64/include benchmarks/src/constrained_qp_regression.cpp \
  /opt/blasfeo_arm64/lib/libblasfeo.a -o build/constrained-release/qp-baseline
python3 - <<'PYQP'
import csv, io, subprocess
from pathlib import Path
root = Path('benchmarks/constrained/results')
programs = {'baseline': 'build/constrained-release/qp-baseline',
            'current': 'build/constrained-release/benchmarks/constrained_qp_regression'}
for name in programs:
    (root / f'qp-{name}.csv').write_text('batch,backend,n,repetition,solve_s\n')
for batch, order in enumerate([('baseline', 'current'), ('current', 'baseline'), ('baseline', 'current')]):
    for name in order:
        rows = csv.reader(io.StringIO(subprocess.check_output([programs[name], '500'], text=True)))
        next(rows)
        with (root / f'qp-{name}.csv').open('a') as out:
            csv.writer(out).writerows([batch, *row] for row in rows)
PYQP
```

## Profile

Use Tracy 0.12.2 to match the pinned client. The following builds local tools without changing PIQP dependencies:

```sh
mkdir -p build/constrained-tools
curl -fL https://github.com/wolfpld/tracy/archive/refs/tags/v0.12.2.zip -o build/constrained-tools/tracy.zip
unzip -qo build/constrained-tools/tracy.zip -d build/constrained-tools
for tracy_tool in capture csvexport; do
  cmake -S "build/constrained-tools/tracy-0.12.2/$tracy_tool" -B "build/constrained-tools/$tracy_tool" -G Ninja -DCMAKE_BUILD_TYPE=Release -DNO_FILESELECTOR=ON -DCPM_SOURCE_CACHE=/tmp/piqp-tracy-deps
  CCACHE_DISABLE=1 cmake --build "build/constrained-tools/$tracy_tool" -j 4
done
cmake -S . -B build/constrained-tracy -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=ON -DBUILD_EXAMPLES=OFF -DBUILD_C_INTERFACE=OFF \
  -DBUILD_WITH_BLASFEO=ON -Dblasfeo_DIR=/opt/blasfeo_arm64 \
  -DBUILD_WITH_TEMPLATE_INSTANTIATION=ON -DBUILD_WITH_TRACY=ON \
  -DFETCHCONTENT_SOURCE_DIR_TRACY="$PWD/build/constrained-tools/tracy-0.12.2"
CCACHE_DISABLE=1 cmake --build build/constrained-tracy --target constrained_solver_benchmark -j 4
mkdir -p build/constrained-profile
build/constrained-tools/capture/tracy-capture -a 127.0.0.1 -f -s 8 -o build/constrained-profile/nql30.tracy &
capture_pid=$!
build/constrained-tracy/benchmarks/constrained_solver_benchmark 20 benchmarks/constrained/instances public benchmarks/constrained/instances/nql30.socp > build/constrained-profile/nql30.csv
wait "$capture_pid"
build/constrained-tools/csvexport/tracy-csvexport -e build/constrained-profile/nql30.tracy > benchmarks/constrained/results/nql30-zones.csv
```

Raw traces stay in `build/constrained-profile`; exclusive zone times stay in `results/nql30-zones.csv`. Instrumented timings are separate from the uninstrumented batches. Two local regularization variants reduced public-problem times by 21% to 27% but were rejected from production because validation covered too few difficult cases: postponing reductions for two iterations after a retry, and keeping the increased regularization as a minimum for the remaining solve. Those exploratory batches used three samples per problem and variant. `results/regularization` retains their source patches and observations.

Generate `results/summary.csv` and the paper's timing table and scaling plots, then build the IEEE paper:

```sh
python3 benchmarks/constrained/summarize.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
```

The output is `paper/main.pdf`. Raw experiment CSVs and environment records remain in `results/`. Any maximum resident size from `/usr/bin/time -l` covers the full process, including input loading, model storage, and the symbolic count observer; it is not solver-only memory.
