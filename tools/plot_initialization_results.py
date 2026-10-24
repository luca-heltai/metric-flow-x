#!/usr/bin/env python3
"""Plot the complete retained initialization/ramp report with deterministic SVG."""
import argparse
import json
from pathlib import Path


def plot(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.labelsize':9,
                                'svg.hashsalt':'metric_flow_initialization_v1',
                                'font.family':'DejaVu Sans','savefig.transparent':False})
    methods=['legacy','native_rest','native_0d']
    colors={'legacy':'#64748b','native_rest':'#007b83','native_0d':'#b86a1b'}
    names={'legacy':'Legacy','native_rest':'Native rest','native_0d':'Native 0D'}
    summaries=report['initialization']['summaries']
    scopes=[('adan56',1,.02),('adan56',2,.02),('network10k',1,.002)]
    labels=['ADAN · MPI 1\n20 ms','ADAN · MPI 2\n20 ms','10k · MPI 1\n2 ms']
    groups={(x['case'],x['ranks'],x['horizon'],x['method']):x for x in summaries}
    assert set(groups)=={(*s,m) for s in scopes for m in methods}
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.5))
    a,b,c,d=axes.ravel();width=.23;x=np.arange(3)
    for j,m in enumerate(methods):
        xx=x+(j-1)*width
        prep=[groups[(*s,m)]['median_phases_s']['preparation_wall_s'] for s in scopes]
        ic=[groups[(*s,m)]['median_phases_s']['ida_setup_and_ic_wall_s'] for s in scopes]
        # Separate markers keep the two measured phases visible across scales;
        # no zero bars or implied absence of native IDA setup.
        a.scatter(xx,prep,c=colors[m],marker='o',s=35,zorder=3)
        a.scatter(xx,ic,c=colors[m],marker='s',s=27,zorder=3)
        for q,pv,iv in zip(xx,prep,ic):a.plot([q,q],[pv,iv],color=colors[m],alpha=.4,lw=1)
        walls=[groups[(*s,m)]['median_process_wall_s'] for s in scopes]
        b.bar(xx,walls,width=.20,color=colors[m],label=names[m])
        for q,value,s in zip(xx,walls,scopes):
            ratio=groups[(*s,m)]['median_paired_process_speedup_vs_legacy']
            text=f'{value:.1f}s' if m=='legacy' else (f'{value:.1f}s\n{ratio:.2f}×' if ratio>=1 else f'{value:.1f}s\n{1/ratio:.2f}× slower')
            b.annotate(text,(q,value),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=7.5)
    a.set_yscale('log');a.set_xticks(x,labels);a.set_ylabel('Wall time (s), logarithmic')
    a.set_title('(a) Preparation and IDA setup/IC',loc='left')
    phase_handles=[Line2D([],[],color='#333333',marker='o',ls='',label='Preparation'),Line2D([],[],color='#333333',marker='s',ls='',label='IDA setup/IC')]
    a.legend(handles=phase_handles,loc='upper left',frameon=False,fontsize=8)
    b.set_xticks(x,labels);b.set_ylabel('Complete process wall time (s)')
    b.set_title('(b) Whole-run medians and paired speed ratios',loc='left')
    b.legend(frameon=False,loc='upper left',fontsize=8);b.margins(y=.30)
    # Colors are shared by phase markers and the whole-process bars.
    a.text(.99,.02,'Colors match panel (b)',transform=a.transAxes,ha='right',va='bottom',fontsize=7.5)
    ramp=report['ramp'];step=ramp['step_counts']
    xx=np.arange(2)
    c.bar(xx-.15,[step['uniform_through_1s'],step['uniform_total']],width=.29,color='#64748b',label='Uniform')
    c.bar(xx+.15,[step['ramp_through_1s'],step['ramp_total']],width=.29,color='#8753a6',label='Ramp')
    for q,y in zip([-.15,.15,.85,1.15],[step['uniform_through_1s'],step['ramp_through_1s'],step['uniform_total'],step['ramp_total']]):
        c.annotate(f'{int(y):,}',(q,y),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    c.set_xticks(xx,['First 1 s','Complete 10 s']);c.set_ylabel('Accepted IDA steps')
    c.set_title('(c) Ramp work and measured speedup',loc='left')
    c.legend(frameon=False,loc='upper left',fontsize=8);c.margins(y=.35)
    c.text(.97,.97,f"Whole process {ramp['performance']['median_paired_process_speedup']:.3f}×\nIDA solve wall {ramp['performance']['median_paired_ida_speedup']:.3f}×\nAll four pairs faster",transform=c.transAxes,ha='right',va='top',fontsize=8)
    metrics=ramp['paired_metric_maxima'];order=['area','velocity','pressure','flow','terminal_area','terminal_velocity','terminal_pressure','terminal_flow','reconstructed_Pc','volume','dVdt','net_flux','imbalance']
    pretty=['A','U','P','Q','Terminal A','Terminal U','Terminal P','Terminal Q','Pc','Volume','dV/dt','Net flux','Imbalance']
    yy=np.arange(len(order))
    d.barh(yy,[metrics[k] for k in order],height=.68,color='#8753a6')
    d.axvline(1,color='#b42318',ls='--',lw=1);d.set_xlim(0,1.05)
    d.set_yticks(yy,pretty,fontsize=7.5);d.invert_yaxis();d.set_xlabel('Maximum normalized late-window error')
    d.set_title('(d) Paired equivalence: every metric < 1',loc='left')
    d.text(.98,.45,f"Reference diagnostic worst:\n{max(ramp['reference_metric_maxima'].values()):.2f} (fails)\nAbsolute accuracy not certified",transform=d.transAxes,ha='right',va='center',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.95})
    for ax in axes.ravel():
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x' if ax is d else 'y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Initialization and startup tolerances · complete process measurements',fontsize=13,y=.985)
    fig.text(.5,.015,'Bars: medians; ratios: median paired speedup vs legacy. Native 0D changes the initial state; its 10k regression is retained.\nDifferent horizons are labeled. 10k MPI 2 is excluded after nonconvergence.\nLate equivalence is against uniform tolerances; reference failures remain diagnostics.',ha='center',va='bottom',fontsize=8)
    fig.subplots_adjust(left=.075,right=.985,top=.91,bottom=.14,hspace=.45,wspace=.30)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,metadata={'Date':None,'Creator':'MetricFlowX plot_initialization_results.py'})
    plt.close(fig)
    if destination.suffix.lower() == '.svg':
        destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=json.loads(a.input.read_text())
    assert report['schema_version']==1
    assert len(report['initialization']['samples'])==27
    assert report['initialization']['complete'] is True
    plot(report,a.output)


if __name__=='__main__':main()
