import csv
import statistics
from collections import defaultdict
from pathlib import Path


root = Path(__file__).resolve().parent
groups = defaultdict(list)
for name in ("piqp-generated", "piqp-public", "clarabel"):
    for row in csv.DictReader((root / "results" / f"{name}.csv").open()):
        groups[row["instance"], row["backend"]].append(row)

summary = {}
for key, rows in sorted(groups.items()):
    times = [float(row["solve_s"]) * 1e6 for row in rows]
    quartiles = statistics.quantiles(times)
    summary[key] = {
        "instance": key[0], "backend": key[1], "samples": len(rows),
        "valid": sum(row["valid"] == "1" for row in rows),
        "native_valid": sum(row.get("native_valid", row["valid"]) == "1" for row in rows),
        "median_us": statistics.median(times), "q1_us": quartiles[0], "q3_us": quartiles[2],
        "setup_us": float(rows[0]["setup_s"]) * 1e6,
        "update_us": statistics.median(float(row.get("update_s", "nan")) * 1e6 for row in rows),
        "factor_us": statistics.median(float(row.get("factor_s", "nan")) * 1e6 for row in rows),
        "iterations": rows[0]["iterations"],
    }
with (root / "results" / "summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=next(iter(summary.values())).keys())
    writer.writeheader()
    writer.writerows(summary.values())

paper = root.parent.parent / "paper"
with (paper / "timings.tex").open("w") as stream:
    stream.write("\\begin{tabular}{lrrrr}\n\\toprule\nInstance & Dense & Sparse & Multi. & Clarabel\\\\\n\\midrule\n")
    for name, label in [("ellipsoid_qcqp_128", "Quadratic, $n=128$"),
                        ("ellipsoid_socp_128", "Cone lift, $n=128$"),
                        ("local_mixed_64", "Mixed, 64 stages"),
                        ("control_80", "Control, $N=80$"),
                        ("nql30", "nql30"), ("nql60", "nql60")]:
        cells = [label]
        for backend in ("dense", "sparse", "multistage", "clarabel"):
            row = summary.get((name, backend))
            cells.append(f"{row['median_us'] / 1000:.3f}" if row else "--")
        stream.write(" & ".join(cells) + "\\\\\n")
    stream.write("\\bottomrule\n\\end{tabular}\n")

with (paper / "scaling.tex").open("w") as stream:
    for family, values, xlabel in [("control", [10, 20, 40, 80], "Horizon $N$"),
                                  ("cone_dimension", [8, 32, 128, 512, 2048], "Cone dimension $d$")]:
        stream.write("\\begin{tikzpicture}\n\\begin{loglogaxis}[width=.48\\textwidth,height=4.6cm,"
                     "xlabel={" + xlabel + "},ylabel={Solve time [$\\mu$s]},grid=major,"
                     "xtick={" + ",".join(map(str, values)) + "},xticklabels={" + ",".join(map(str, values)) + "},"
                     "legend style={font=\\scriptsize},legend pos=north west]\n")
        for backend, label in [("dense", "Dense"), ("sparse", "Sparse"),
                               ("multistage", "Multistage"), ("clarabel", "Clarabel")]:
            coordinates = " ".join(f"({n},{summary[family + '_' + str(n), backend]['median_us']:.6g})" for n in values)
            stream.write("\\addplot+[mark size=1.5pt] coordinates {" + coordinates + "};\n")
            stream.write("\\addlegendentry{" + label + "}\n")
        stream.write("\\end{loglogaxis}\n\\end{tikzpicture}\n")
