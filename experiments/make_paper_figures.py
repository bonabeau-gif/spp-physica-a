from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / 'results'
FIG = ROOT / 'figures' / 'paper'
FIG.mkdir(parents=True, exist_ok=True)

from spp_unified.parameters import (
    ACTIVE_SINGLE_NAMES, SingleParams, TwoSpeciesParams, SimulationConfig
)
from spp_unified.model import simulate_two_species


def panel_label(ax, label):
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontweight='bold',
            fontsize=11, va='bottom', ha='left')


def archive_grid(df, dims, xlim, ylim, value='objective'):
    Z = np.full(dims, np.nan)
    for _, row in df.iterrows():
        i, j = int(row.cell_0), int(row.cell_1)
        if 0 <= i < dims[0] and 0 <= j < dims[1]:
            Z[j, i] = row[value]
    return Z


def make_single_archives():
    a = pd.read_csv(RES / 'single_qd_alignment_milling.csv')
    b = pd.read_csv(RES / 'single_qd_alignment_clustering.csv')
    dims = (12, 12)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), constrained_layout=True)
    z1 = archive_grid(a, dims, (0,1), (0,.5))
    z2 = archive_grid(b, dims, (0,1), (0,8))
    im1 = axes[0].imshow(z1, origin='lower', extent=(0,1,0,.5), aspect='auto',
                         vmin=.80, vmax=1.0)
    im2 = axes[1].imshow(z2, origin='lower', extent=(0,1,0,8), aspect='auto',
                         vmin=.80, vmax=1.0)
    axes[0].set(xlabel='polar alignment', ylabel='milling')
    axes[1].set(xlabel='polar alignment', ylabel='clustering')
    panel_label(axes[0], 'a'); panel_label(axes[1], 'b')
    cbar = fig.colorbar(im2, ax=axes, shrink=.88, pad=.02)
    cbar.set_label('temporal stability')
    fig.savefig(FIG / 'fig_single_archives.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_single_method_robustness():
    de = pd.read_csv(RES / 'de_vs_qd_budgeted.csv')
    fs = pd.read_csv(RES / 'single_finite_size.csv')
    imp = pd.read_csv(RES / 'single_oat_importance.csv')
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)

    # Method comparison.
    objectives = ['alignment','milling','clustering']
    x = np.arange(len(objectives))
    methods=['DE (144)','QD (160)','QD pooled (640)']
    for k, method in enumerate(methods):
        vals = [float(de[(de.objective==o)&(de.method==method)].best.iloc[0]) for o in objectives]
        axes[0].bar(x + (k-1)*0.25, vals, width=0.24, label=method)
    axes[0].set_xticks(x, objectives)
    axes[0].set_ylabel('best objective value')
    axes[0].set_ylim(0,1.06)
    axes[0].legend(frameon=False, fontsize=8)
    panel_label(axes[0], 'a')

    # Finite size.
    for state, marker in [('aligned','o'),('milling','s'),('clustered','^')]:
        raw=fs[fs.state==state]
        metric = {'aligned':'phi','milling':'milling','clustered':'clustering'}[state]
        g=raw.groupby('S')[metric].agg(['mean','std']).reset_index().sort_values('S')
        axes[1].errorbar(g.S, g['mean'], yerr=g['std'], marker=marker,
                         capsize=3, label=state)
    axes[1].set_xlabel('box size, S')
    axes[1].set_ylabel('state-specific order parameter')
    axes[1].legend(frameon=False, fontsize=8)
    panel_label(axes[1], 'b')

    # Sensitivity heat map.
    piv = imp.pivot(index='parameter', columns='state', values='range')
    pretty = {
        'rho':'density', 'v0':'speed', 'inertia':'inertia',
        'interaction_range_ratio':'interaction range', 'force_strength':'force strength',
        'adhesion':'adhesion', 'alignment_range_ratio':'alignment range',
        'alignment_strength':'alignment strength', 'noise':'noise'
    }
    order = ['noise','alignment_strength','force_strength','alignment_range_ratio',
             'interaction_range_ratio','adhesion','v0','rho','inertia']
    piv=piv.reindex(order)
    im=axes[2].imshow(piv[['aligned','milling']].to_numpy(), aspect='auto', vmin=0, vmax=1)
    axes[2].set_xticks([0,1], ['aligned','milling'])
    axes[2].set_yticks(np.arange(len(order)), [pretty[q] for q in order], fontsize=8)
    cb=fig.colorbar(im, ax=axes[2], shrink=.8, pad=.02)
    cb.set_label('range across sweep')
    panel_label(axes[2], 'c')
    fig.savefig(FIG / 'fig_single_method_robustness.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_multi_archives():
    rec = pd.read_csv(RES / 'multi_qd_reciprocal.csv')
    non = pd.read_csv(RES / 'multi_qd_nonreciprocal.csv')
    dims=(12,12)
    fig, axes=plt.subplots(1,2,figsize=(8.2,3.7),constrained_layout=True)
    zrec=archive_grid(rec,dims,(0,1),(0,1)); znon=archive_grid(non,dims,(0,1),(0,1))
    im1=axes[0].imshow(zrec,origin='lower',extent=(0,1,0,1),aspect='auto',vmin=.80,vmax=1.0)
    im2=axes[1].imshow(znon,origin='lower',extent=(0,1,0,1),aspect='auto',vmin=.80,vmax=1.0)
    for ax in axes:
        ax.set_xlabel('net center-of-mass drift')
        ax.set_ylabel('persistent chirality score')
    axes[0].text(.03,.96,'reciprocal',transform=axes[0].transAxes,va='top')
    axes[1].text(.03,.96,'non-reciprocal',transform=axes[1].transAxes,va='top')
    panel_label(axes[0],'a'); panel_label(axes[1],'b')
    cb=fig.colorbar(im2,ax=axes,shrink=.88,pad=.02);cb.set_label('temporal stability')
    fig.savefig(FIG/'fig_multi_archives.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def non_params_from_row(row):
    base=SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES]))
    return TwoSpeciesParams(base=base, fraction_B=float(row.fraction_B),
        alignment_symmetric=float(row.alignment_symmetric),
        alignment_antisymmetric=float(row.alignment_antisymmetric),
        force_symmetric=float(row.force_symmetric),
        force_antisymmetric=float(row.force_antisymmetric))


def global_heading(result):
    th=result.timeseries['theta']
    z=np.mean(np.exp(1j*th),axis=1)
    return np.unwrap(np.angle(z)), np.abs(z)


def make_chirality_validation():
    seedvals=pd.read_csv(RES/'multi_chirality_seed_values.csv')
    hor=pd.read_csv(RES/'multi_chirality_horizon.csv')
    non=pd.read_csv(RES/'multi_qd_nonreciprocal.csv')
    row=non.loc[non.chirality.idxmax()]
    pn=non_params_from_row(row)
    pc=replace(pn, alignment_antisymmetric=0.0, force_antisymmetric=0.0)
    cfg=SimulationConfig(steps=800,sample_start=400,sample_every=2,boundary='periodic',seed=0)
    rn=simulate_two_species(pn,cfg)
    rc=simulate_two_species(pc,cfg)
    an, phin=global_heading(rn); ar, phir=global_heading(rc)
    t=np.arange(len(an))*cfg.sample_every

    fig,axes=plt.subplots(1,3,figsize=(11.0,3.55),constrained_layout=True)
    order=['nonreciprocal_chiral','matched_reciprocal','best_reciprocal_search']
    labels=['non-reciprocal\nchiral','matched\nreciprocal','best reciprocal\nsearch elite']
    vals=[seedvals[seedvals.condition==q].chirality.to_numpy() for q in order]
    axes[0].boxplot(vals,tick_labels=labels,showmeans=True)
    axes[0].set_ylabel('persistent chirality score')
    axes[0].set_ylim(-.03,1.03)
    panel_label(axes[0],'a')

    for label,g in hor.groupby('condition',sort=False):
        axes[1].errorbar(g.steps,g.chirality,yerr=g.chirality_sd,marker='o',capsize=3,
                         label=label.replace('_',' '))
    axes[1].set_xlabel('simulation steps')
    axes[1].set_ylabel('persistent chirality score')
    axes[1].set_ylim(-.03,1.03)
    axes[1].legend(frameon=False,fontsize=7)
    panel_label(axes[1],'b')

    axes[2].plot(t,an,label='non-reciprocal')
    axes[2].plot(t,ar,label='matched reciprocal')
    axes[2].set_xlabel('time step')
    axes[2].set_ylabel('unwrapped global heading (rad)')
    axes[2].legend(frameon=False,fontsize=8)
    panel_label(axes[2],'c')
    fig.savefig(FIG/'fig_chirality_validation.png',dpi=300,bbox_inches='tight')
    plt.close(fig)


def make_coverage():
    paths={
        'single alignment-milling':'single_qd_alignment_milling_history.csv',
        'single alignment-clustering':'single_qd_alignment_clustering_history.csv',
        'two species, reciprocal':'multi_qd_reciprocal_history.csv',
        'two species, non-reciprocal':'multi_qd_nonreciprocal_history.csv',
    }
    fig,ax=plt.subplots(figsize=(5.4,3.7))
    for label,f in paths.items():
        g=pd.read_csv(RES/f)
        ax.plot(g.evaluations,g.coverage,marker='o',markersize=3,label=label)
    ax.set_xlabel('simulator evaluations')
    ax.set_ylabel('archive coverage')
    ax.legend(frameon=False,fontsize=7)
    fig.tight_layout();fig.savefig(FIG/'fig_archive_coverage.png',dpi=300,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    make_single_archives()
    make_single_method_robustness()
    make_multi_archives()
    make_chirality_validation()
    make_coverage()
    print('wrote', sorted(p.name for p in FIG.glob('*.png')))
