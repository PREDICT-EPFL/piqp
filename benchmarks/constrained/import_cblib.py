# /// script
# requires-python = ">=3.11"
# ///
import argparse
import gzip
import json
from pathlib import Path


def convert(source, destination):
    lines = iter(line.strip() for line in gzip.decompress(source.read_bytes()).decode().splitlines()
                 if line.strip() and not line.startswith("#"))
    entries, objective, rhs, blocks = [], {}, {}, []
    n = rows = 0
    for section in lines:
        if section == "VER":
            assert next(lines) == "1"
        elif section == "OBJSENSE":
            assert next(lines) == "MIN"
        elif section in ("VAR", "CON"):
            count, count_blocks = map(int, next(lines).split())
            parsed = [(kind, int(size)) for kind, size in (next(lines).split() for _ in range(count_blocks))]
            if section == "VAR":
                n = count
                assert all(kind == "F" for kind, _ in parsed), "only free continuous variables are supported"
            else:
                rows, blocks = count, parsed
        elif section == "OBJACOORD":
            for _ in range(int(next(lines))):
                i, value = next(lines).split()
                objective[int(i)] = float(value)
        elif section == "ACOORD":
            for _ in range(int(next(lines))):
                i, j, value = next(lines).split()
                entries.append((int(i), int(j), float(value)))
        elif section == "BCOORD":
            for _ in range(int(next(lines))):
                i, value = next(lines).split()
                rhs[int(i)] = float(value)
        else:
            raise ValueError(f"unsupported CBF section {section}")
    row_entries = [[] for _ in range(rows)]
    for i, j, value in entries:
        row_entries[i].append((j, value))
    eq, b, linear, h, cones = [], [], [], [], []
    offset = 0
    for kind, size in blocks:
        if kind in ("L=", "L+", "L-"):
            target, values = (eq, b) if kind == "L=" else (linear, h)
            sign = -1 if kind == "L+" else 1
            for i in range(offset, offset + size):
                row = len(values)
                target.extend((row, j, sign * value) for j, value in row_entries[i])
                values.append(-sign * rhs.get(i, 0))
        elif kind in ("Q", "QR"):
            ids = sorted({j for i in range(offset, offset + size) for j, _ in row_entries[i]})
            positions = {j: k for k, j in enumerate(ids)}
            F = [(i-offset, positions[j], value) for i in range(offset, offset+size) for j, value in row_entries[i]]
            cones.append((kind == "QR", size, ids, F, [rhs.get(i, 0) for i in range(offset, offset+size)]))
        else:
            raise ValueError(f"unsupported CBF cone {kind}")
        offset += size
    assert offset == rows
    with destination.open("w") as out:
        def vector(v):
            out.write(" ".join(format(x, ".17g") for x in v) + "\n")

        def matrix(r, c, coordinates):
            out.write(f"{r} {c} {len(coordinates)}\n")
            for i, j, value in coordinates:
                out.write(f"{i} {j} {value:.17g}\n")

        out.write(f"{n}\n")
        vector([objective.get(i, 0) for i in range(n)])
        matrix(len(b), n, eq)
        vector(b)
        matrix(len(h), n, linear)
        vector(h)
        out.write(f"{len(cones)}\n")
        for rotated, size, ids, F, f in cones:
            out.write(f"{int(rotated)} {len(ids)}\n")
            vector(ids)
            matrix(size, len(ids), F)
            vector(f)
    return {"instance": source.name, "n": n, "equalities": len(b), "inequalities": len(h), "cones": len(cones)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        print(json.dumps(convert(source, args.output / (source.name.removesuffix(".cbf.gz") + ".socp"))))


if __name__ == "__main__":
    main()
