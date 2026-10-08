# Generates multistage_partitioning.svg illustrating the partitioning of the
# sparse_multistage_parallel KKT solver.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# 13 stage blocks split into 3 segments (5 | sep | 3 | sep | 3) plus global block
segments = [[0, 1, 2, 3, 4], [6, 7, 8], [10, 11, 12]]
separators = [5, 9]
n_stages = 13
G = n_stages  # index of global block

seg_of = {i: k for k, seg in enumerate(segments) for i in seg}
serial = set(separators) | {G}

colors = ['#4C72B0', '#DD8452', '#55A868']
serial_color = '#8C8C8C'


def nonzero(i, j):
    return i == G or j == G or abs(i - j) <= 1


def fill_in(i, j):
    # coupling of segment k > 0 with its preceding separator becomes dense
    for k in range(1, len(segments)):
        sep = separators[k - 1]
        if (i == sep and j in segments[k]) or (j == sep and i in segments[k]):
            return not nonzero(i, j)
    # consecutive separators get coupled through the segment in between
    if i in separators and j in separators:
        return i != j
    return False


def block_color(i, j):
    if i in serial and j in serial:
        return serial_color
    return colors[seg_of[i] if i not in serial else seg_of[j]]


def draw(ax, order, parallel):
    pos = {b: p for p, b in enumerate(order)}
    n = len(order)
    for i in order:
        for j in order:
            x, y = pos[j], pos[i]
            if nonzero(i, j):
                color = block_color(i, j) if parallel else serial_color
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor='white', lw=1))
            elif parallel and fill_in(i, j):
                # inset to match the visible size of the nonzero blocks
                d = 0.04
                ax.add_patch(Rectangle((x + d, y + d), 1 - 2 * d, 1 - 2 * d, facecolor='white',
                                       edgecolor=block_color(i, j), hatch='////', lw=1))
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#cccccc')


fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))

original = list(range(n_stages)) + [G]
draw(axes[0], original, False)
axes[0].set_title('Original KKT matrix $\\Psi$')

permuted = [i for seg in segments for i in seg] + separators + [G]
draw(axes[1], permuted, True)
axes[1].set_title('Permuted KKT matrix $P \\Psi P^\\top$')

# outline the sequential reduced system
n_par = sum(len(s) for s in segments)
n = len(permuted)
axes[1].add_patch(Rectangle((n_par, n_par), n - n_par, n - n_par, fill=False, edgecolor='black', lw=2, ls='--',
                           clip_on=False, zorder=3))

legend = [Patch(facecolor=c, label=f'thread {k + 1} (parallel)') for k, c in enumerate(colors)]
legend.append(Patch(facecolor=serial_color, label='sequential'))
legend.append(Patch(facecolor='white', edgecolor='black', hatch='////', label='fill-in'))
fig.legend(handles=legend, loc='lower center', ncol=5, frameon=False)

fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig('multistage_partitioning.svg', bbox_inches='tight')
