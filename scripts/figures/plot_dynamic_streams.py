"""Dynamic-graph (real-timestamp) comparison figures for the SAG paper.

Data inlined from results/dynamic/all_metrics_dynamic.csv
(strict single-pass temporal stream,
AUPRC / Macro-F1 use the balanced CoLA/DOMINANT protocol).

Outputs:
  fig_dyn_auroc_bars.{pdf,png}   (a)-(c) per-dataset AUROC, SAG vs. baselines
  fig_dyn_sag_gain.{pdf,png}     SAG vs. strongest baseline on all metrics
  fig_dyn_neo4j.{pdf,png}        Neo4j system-level streaming (all_metrics_neo4j.csv)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 22
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['axes.linewidth'] = 3

# ================== Data (%) : (AUROC, AUPRC_bal, Macro-F1_bal) ==================
DATASETS = ['AMLSim', 'Bitcoin-Alpha', 'Bitcoin-OTC']
RESULTS = {
    'AMLSim': {
        'SAG':       (99.849, 99.644, 99.570),
        'JODIE':     (98.626, 99.245, 98.594),
        'EvolveGCN': (93.051, 93.422, 82.724),
        'MIDAS':     (71.671, 75.158, 70.221),
        'TGAT':      (55.456, 66.243, 59.380),
        'TGN':       (0.104, 31.078, 33.333),
        'DyRep':     (0.000, 31.074, 33.333),
    },
    'Bitcoin-Alpha': {
        'SAG':       (65.554, 59.284, 65.275),
        'MIDAS':     (54.518, 53.934, 45.245),
        'JODIE':     (42.664, 42.932, 49.426),
        'EvolveGCN': (41.751, 43.458, 46.038),
        'TGAT':      (38.070, 40.561, 46.997),
        'DyRep':     (31.075, 37.343, 42.430),
        'TGN':       (29.039, 37.182, 39.991),
        'SedanSpot': (16.958, 33.663, 34.645),
    },
    'Bitcoin-OTC': {
        'SAG':       (66.407, 59.578, 64.552),
        'MIDAS':     (53.912, 53.120, 43.680),
        'JODIE':     (47.609, 45.241, 50.636),
        'EvolveGCN': (41.740, 44.940, 44.970),
        'DyRep':     (40.985, 42.028, 49.415),
        'TGAT':      (30.041, 37.401, 40.031),
        'TGN':       (29.490, 37.227, 39.071),
        'SedanSpot': (16.194, 33.529, 34.654),
    },
}

SAG_COLOR = '#d61737'
BASE_COLOR = '#3634f2'
TXT_BASE = '#1a1a8c'
METRICS = ['AUROC', 'AUPRC$_\\mathrm{bal}$', 'Macro-F1$_\\mathrm{bal}$']
METRIC_COLORS = ['#d61737', '#3634f2', '#ffa810']


def save(fig, name):
    fig.savefig(name + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(name + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}.{{pdf,png}}')


def style(ax):
    ax.grid(True, axis='y', alpha=0.7, linestyle='--', linewidth=3)
    ax.set_axisbelow(True)
    ax.tick_params(width=3, length=10)


# ================== Figure A: per-dataset AUROC ==================
def fig_auroc_bars():
    fig, axes = plt.subplots(1, 3, figsize=(24, 8.5), sharey=True)
    letters = 'abc'
    for di, (ax, ds) in enumerate(zip(axes, DATASETS)):
        methods = list(RESULTS[ds].keys())          # already sorted, SAG first
        vals = [RESULTS[ds][m][0] for m in methods]
        colors = [SAG_COLOR if m == 'SAG' else BASE_COLOR for m in methods]
        x = np.arange(len(methods))
        bars = ax.bar(x, vals, 0.7, color=colors, alpha=0.88,
                      edgecolor='black', linewidth=2.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.1f}',
                    rotation=90, ha='center', va='bottom', fontweight='bold',
                    fontsize=20, color=SAG_COLOR if b.get_facecolor()[0] > 0.5 else TXT_BASE)
        ax.axhline(50, color='black', linestyle=':', linewidth=2.5, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=60, ha='right', rotation_mode='anchor')
        ax.get_xticklabels()[0].set_color(SAG_COLOR)
        ax.set_ylim(0, 120)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        style(ax)
        if di == 0:
            ax.set_ylabel('AUROC (%)', fontsize=30, fontweight='bold')
        ax.text(0.5, -0.36, f'({letters[di]}) {ds}', ha='center', va='top',
                fontweight='bold', fontsize=30, transform=ax.transAxes)
    handles = [Patch(facecolor=SAG_COLOR, edgecolor='black', label='SAG (Ours)'),
               Patch(facecolor=BASE_COLOR, edgecolor='black', label='Baselines'),
               plt.Line2D([0], [0], color='black', linestyle=':', linewidth=2.5,
                          label='Random')]
    leg = axes[2].legend(handles=handles, loc='upper right', frameon=True,
                         fancybox=True, shadow=True, fontsize=22)
    leg.get_frame().set_linewidth(3)
    leg.get_frame().set_edgecolor('black')
    plt.subplots_adjust(bottom=0.30, left=0.06, right=0.99, top=0.97, wspace=0.08)
    save(fig, 'fig_dyn_auroc_bars')


# ================== Figure B: SAG vs. strongest baseline ==================
def fig_sag_gain():
    fig, ax = plt.subplots(figsize=(20, 8.5))
    n_m = len(METRICS)
    bar_w = 0.13
    group_x = np.arange(len(DATASETS)) * 1.1
    for di, ds in enumerate(DATASETS):
        base = max((m for m in RESULTS[ds] if m != 'SAG'),
                   key=lambda m: RESULTS[ds][m][0])
        for mi in range(n_m):
            for si, (who, hatch, alpha) in enumerate([('SAG', None, 0.92),
                                                      (base, '//', 0.45)]):
                k = mi * 2 + si
                xpos = group_x[di] + (k - (2 * n_m - 1) / 2.0) * bar_w
                v = RESULTS[ds][who][mi]
                ax.bar(xpos, v, bar_w, color=METRIC_COLORS[mi], alpha=alpha,
                       hatch=hatch, edgecolor='black', linewidth=2.2)
                ax.text(xpos, v + 1.2, f'{v:.1f}', rotation=90, ha='center',
                        va='bottom', fontweight='bold', fontsize=19)
    ax.set_xticks(group_x)
    ax.set_xticklabels(DATASETS, fontsize=26)
    ax.set_ylim(30, 118)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_ylabel('Score (%)', fontsize=30, fontweight='bold')
    style(ax)
    metric_h = [Patch(facecolor=c, edgecolor='black', alpha=0.92, label=m)
                for m, c in zip(METRICS, METRIC_COLORS)]
    who_h = [Patch(facecolor='white', edgecolor='black', label='SAG'),
             Patch(facecolor='white', edgecolor='black', hatch='//',
                   label='Best baseline')]
    leg1 = ax.legend(handles=metric_h, loc='upper left', ncol=3, frameon=True,
                     fancybox=True, shadow=True, fontsize=21)
    leg1.get_frame().set_linewidth(3)
    leg1.get_frame().set_edgecolor('black')
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=who_h, loc='upper right', ncol=2, frameon=True,
                     fancybox=True, shadow=True, fontsize=21)
    leg2.get_frame().set_linewidth(3)
    leg2.get_frame().set_edgecolor('black')
    plt.subplots_adjust(bottom=0.10, left=0.07, right=0.99, top=0.97)
    save(fig, 'fig_dyn_sag_gain')


# ================== Figure C: Neo4j system-level streaming ==================
# edge_limit=100k, mean of 2 runs: (AUROC %, query+score latency ms)
NEO4J = {
    'AMLSim':        {'SAG': (99.849, 5399.05), 'JODIE': (97.85, 21913.70), 'MIDAS': (71.62, 44.74)},
    'Bitcoin-Alpha': {'SAG': (65.554, 4801.76), 'JODIE': (46.20, 12479.07), 'MIDAS': (55.42, 37.43)},
}
NEO4J_COLORS = {'SAG': '#d61737', 'JODIE': '#3634f2', 'MIDAS': '#ffa810'}


def fig_neo4j():
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    methods = ['SAG', 'JODIE', 'MIDAS']
    dsets = list(NEO4J.keys())
    bar_w = 0.36
    x = np.arange(len(methods))
    for pi, ax in enumerate(axes):
        for di, ds in enumerate(dsets):
            off = (di - 0.5) * bar_w
            for mi, m in enumerate(methods):
                v = NEO4J[ds][m][pi]
                ax.bar(x[mi] + off, v, bar_w, color=NEO4J_COLORS[m],
                       alpha=0.92 if di == 0 else 0.5, edgecolor='black',
                       linewidth=2.2, hatch=None if di == 0 else '//')
                lab = f'{v:.1f}' if pi == 0 else f'{v / 1000:.1f}s' if v >= 1000 else f'{v:.0f}ms'
                ypos = v + 1.5 if pi == 0 else v * 1.15
                ax.text(x[mi] + off, ypos, lab, rotation=90, ha='center',
                        va='bottom', fontweight='bold', fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(['SAG (Ours)', 'JODIE', 'MIDAS'], fontsize=24)
        style(ax)
        if pi == 0:
            ax.set_ylim(30, 118)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.set_ylabel('AUROC (%)', fontsize=30, fontweight='bold')
            title = '(a) AUROC'
        else:
            ax.set_yscale('log')
            ax.set_ylim(10, 3e5)
            ax.set_ylabel('Latency (ms)', fontsize=30, fontweight='bold')
            title = '(b) Latency'
        ax.text(0.5, -0.12, title, ha='center', va='top', fontweight='bold',
                fontsize=28, transform=ax.transAxes)
    hs = [Patch(facecolor='white', edgecolor='black', label='AMLSim'),
          Patch(facecolor='white', edgecolor='black', hatch='//',
                label='Bitcoin-Alpha')]
    leg = fig.legend(handles=hs, loc='upper center', ncol=2, frameon=True,
                     fancybox=True, shadow=True, fontsize=24,
                     bbox_to_anchor=(0.5, 1.03))
    leg.get_frame().set_linewidth(3)
    leg.get_frame().set_edgecolor('black')
    plt.subplots_adjust(bottom=0.16, left=0.07, right=0.99, top=0.88, wspace=0.22)
    save(fig, 'fig_dyn_neo4j')


if __name__ == '__main__':
    fig_auroc_bars()
    fig_sag_gain()
    fig_neo4j()
