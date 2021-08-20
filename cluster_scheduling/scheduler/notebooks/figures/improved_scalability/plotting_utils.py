from matplotlib import pyplot as plt
import matplotlib; matplotlib.font_manager._rebuild()
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from matplotlib import rc
import numpy as np
sns.set_style('ticks')
font = {
    'font.family':'Roboto',
    'font.size': 12,
}
sns.set_style(font)
paper_rc = {
    'lines.linewidth': 3,
    'lines.markersize': 10,
}
sns.set_context("paper", font_scale=2, rc=paper_rc)
current_palette = sns.color_palette()


def plot_runtimes(data, xticks, yticks, yticklabels, output_filename=None):
    plt.figure(figsize=(6.5, 3))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    sns.lineplot(x='num_jobs', y='runtimes', style='policy',
                 hue='policy',
                 data=data, ci=None,
                 markers=True)
    ax.set_xlabel("Number of jobs")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_xlim([min(xticks), max(xticks)])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    sns.despine()

    leg = plt.legend(loc='upper left', frameon=False)
    
    if output_filename is not None:
        with PdfPages(output_filename) as pdf:
            pdf.savefig(bbox_inches='tight')
            
    plt.show()
    

def plot_effective_throughput_ratios(all_effective_throughputs, cdf=False,
                                     output_filename=None):
    plt.figure(figsize=(6.5, 3))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    job_ids = list(all_effective_throughputs[0].keys())
    for (effective_throughputs, label) in zip(
        all_effective_throughputs[1:], ['4 sub-clusters', '16 sub-clusters']):
        effective_throughput_ratios = [
            effective_throughputs[job_id] / all_effective_throughputs[0][job_id]
            for job_id in job_ids]
        if cdf:
            ax.plot([float((i+1) * 100.0) / len(effective_throughputs)
                     for i in range(len(effective_throughputs))],
                    sorted(effective_throughput_ratios), label=label)
        else:
            ax.plot(range(len(effective_throughputs)),
                    effective_throughput_ratios, label=label)
    ax.axhline(1.0, color='k', linestyle=':', linewidth=2)
    if cdf:
        ax.set_xlabel("CDF (%)")
    else:
        ax.set_xlabel("Job ID")
    ax.set_ylabel("Effective\nthroughput ratio")
    if cdf:
        ax.set_xlim([0, 100.0])
    else:
        ax.set_xlim([0, len(effective_throughputs)])
    ax.set_ylim([0, 2.0])
    plt.legend(frameon=False)
    sns.despine()
    
    if output_filename is not None:
        with PdfPages(output_filename) as pdf:
            pdf.savefig(bbox_inches='tight')
            
    plt.show()


def plot_runtime_vs_effective_throughput_ratios(runtimes, all_effective_throughputs, labels,
                                                draw_arrow=False,
                                                output_filename=None):
    plt.figure(figsize=(6.5, 3))
    ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    job_ids = list(all_effective_throughputs[0].keys())
    for (runtime, effective_throughputs, label) in zip(
        runtimes,
        all_effective_throughputs,
        labels):
        effective_throughput_ratios = np.array([
            effective_throughputs[job_id] / all_effective_throughputs[0][job_id]
            for job_id in job_ids])
        mean = np.mean(effective_throughput_ratios)
        print(label, runtime, mean, np.std(effective_throughput_ratios))
        ax.scatter(runtime, mean, label=label)
        ax.errorbar(runtime, mean, np.std(effective_throughput_ratios))
        ax.annotate(label, (runtime*1.15, mean-0.08))

    ax.set_ylabel("Throughput ratio")
    ax.set_xlabel("Runtime (seconds)")
    sns.despine()
    
    ax.set_xscale('log')
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlim(xmin, xmax*1.2)
    plt.ylim(0, ymax*1.15)
    
    if draw_arrow:
        ax.annotate('Ideal', xy=(300, 0.8), xytext=(900, 0.3), 
            arrowprops=dict(facecolor='black', shrink=0.),
        )
    
    if output_filename is not None:
        with PdfPages(output_filename) as pdf:
            pdf.savefig(bbox_inches='tight')
            
    plt.show()