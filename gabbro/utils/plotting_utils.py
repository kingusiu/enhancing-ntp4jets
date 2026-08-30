import os
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import seaborn as sns
import mplhep as hep
import pandas as pd


def make_pairwise_heatmaps(data, num_dim):
    heatmaps = []
    xedges = []
    yedges = []

    for i in range(0, num_dim, 2):
        heatmap, xedge, yedge = np.histogram2d(data[:, i], data[:, i+1], bins=70, normed=True)
        heatmaps.append(heatmap)
        xedges.append(xedge)
        yedges.append(yedge)

    return np.asarray(heatmaps), np.asarray(xedges), np.asarray(yedges)


def calculate_nrows_ncols(latent_dim_n):
    # calculate number of subplots on canvas
    nrows = int(round(math.sqrt(latent_dim_n/2)))
    ncols = math.ceil(math.sqrt(latent_dim_n/2))
    return nrows, ncols


def plot_latent_space_2D(latent_coords, title_suffix="data", filename_suffix="data", fig_dir='fig'): # ndarray [M x K] 
    '''
        plot latent space coordinates by slices of 2D
        latent_coords ... ndarray of sim [M x K] containing M samples with each K latent coordinates
    '''

    latent_dim_n = latent_coords.shape[1] - 1 if latent_coords.shape[1] % 2 else latent_coords.shape[1] # if num latent dims is odd, slice off last dim

    heatmaps, xedges, yedges = make_pairwise_heatmaps(latent_coords, latent_dim_n)    

    min_hist_val = np.min(heatmaps[heatmaps > 0])  # find min value for log color bar clipping zero values
    max_hist_val = np.max(heatmaps)

    extent = [np.min(xedges), np.max(xedges), np.min(yedges), np.max(yedges)]

    # calculate number of subplots on canvas
    nrows, ncols = calculate_nrows_ncols(latent_dim_n)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 9))

    for d, ax in zip(range(len(heatmaps)), axs.flat[:latent_dim_n]):
        im = ax.imshow(heatmaps[d].T, extent=extent, origin='lower', norm=colors.LogNorm(vmin=min_hist_val, vmax=max_hist_val))
        ax.set_title('dims {} & {}'.format(d*2, d*2+1), fontsize='small')
        # fig.colorbar(im, ax=ax)

    for a in axs.flat: a.set_xticks(a.get_xticks()[::2])
    if axs.size > latent_dim_n/2:
        for a in axs.flat[int(latent_dim_n/2):]: a.axis('off')

    plt.suptitle('latent space ' + title_suffix)
    cb = fig.colorbar(im)
    cb.set_label('count')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'latent_space_2D_hist_' + filename_suffix + '.png'))
    plt.close(fig)




def plot_latent_space_2D_bg_vs_sig(latent_bg, latent_sig, title_suffix=None, filename_suffix=None, fig_dir='fig', contour_labels=False):
    '''
        plot latent space coordinates by slices of 2D
        latent_coords ... ndarray of sim [M x K] containing M samples with each K latent coordinates
    '''

    latent_dim_n = latent_bg.shape[1] - 1 if latent_bg.shape[1] % 2 else latent_bg.shape[1] # if num latent dims is odd, slice off last dim

    heatmaps_bg, xedges_bg, yedges_bg = make_pairwise_heatmaps(latent_bg, latent_dim_n)
    heatmaps_sig, xedges_sig, yedges_sig = make_pairwise_heatmaps(latent_sig, latent_dim_n)

    min_hist_val = min(np.min(heatmaps_bg[heatmaps_bg > 0]), np.min(heatmaps_sig[heatmaps_sig > 0]))  # find min value for log color bar clipping zero values
    max_hist_val = max(np.max(heatmaps_bg), np.max(heatmaps_sig))

    extent = [min(np.min(xedges_bg), np.min(xedges_sig)), max(np.max(xedges_bg), np.max(xedges_sig)), \
              min(np.min(yedges_bg), np.min(yedges_sig)), max(np.max(yedges_bg), np.max(yedges_sig))]

    # calculate number of subplots on canvas
    nrows, ncols = calculate_nrows_ncols(latent_dim_n)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 9), sharex=False, sharey=False)
    # make dark controur lines
    levels_n = 7
    colors_bg = [mpl.colors.rgb2hex(cm.get_cmap('Blues')(int(i))) for i in np.linspace(60, 320, levels_n)]
    colors_sig = [mpl.colors.rgb2hex(cm.get_cmap('Oranges')(int(i))) for i in np.linspace(60, 320, levels_n)]

    for d, (heat_bg, heat_sig, ax) in enumerate(zip(heatmaps_bg, heatmaps_sig, axs.flat[:latent_dim_n])):
        cont_bg = ax.contour(heat_bg.T, cmap=cm.get_cmap('Blues')) #, norm=colors.LogNorm(), colors=colors_bg, extent=extent, levels=levels_n)
        cont_sig = ax.contour(heat_sig.T, cmap=cm.get_cmap('Oranges')) #, norm=colors.LogNorm(), colors=colors_sig, extent=extent, levels=levels_n)
        ax.set_title('dims {} & {}'.format(d*2+1, d*2+2), fontsize='small')
        if contour_labels:
            ax.clabel(cont_bg, colors='k', fontsize=5.)
            ax.clabel(cont_sig, colors='k', fontsize=5.)
        
    # fig.colorbar(cont_bg, ax=axs.flat[-1])
    # fig.colorbar(cont_sig, ax=axs.flat[-1])
        
    for a in axs.flat: a.set_xticks(a.get_xticks()[::2])
    if axs.size > latent_dim_n/2:
        for a in axs.flat[int(latent_dim_n/2):]: a.axis('off')

    plt.legend([cont_bg.collections[0], cont_sig.collections[0]], ['Background', 'Signal'], loc='center')
    plt.suptitle(' '.join(filter(None, ['latent space ', title_suffix])))
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, '_'.join(filter(None, ['latent_space_2D_contour', filename_suffix, '.png']))))
    plt.close(fig)


def plot_clusters(latent_coords, cluster_assignemnts, cluster_centers=None, title_suffix=None, filename_suffix=None, fig_dir='fig'):

    latent_dim_n = latent_coords.shape[1] - 1 if latent_coords.shape[1] % 2 else latent_coords.shape[1] # if num latent dims is odd, slice off last dim
    nrows, ncols = calculate_nrows_ncols(latent_dim_n)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(9, 9))

    for d, ax in zip(range(0, latent_dim_n, 2), axs.flat):
        ax.scatter(latent_coords[:,d], latent_coords[:,d+1], c=cluster_assignemnts, s=1.5, cmap='tab10')
        ax.set_title('dims {} & {}'.format(d+1, d+2), fontsize='small')
        if cluster_centers is not None:
            ax.scatter(cluster_centers[:, d], cluster_centers[:, d+1], c='black', s=100, alpha=0.7);

    if axs.size > latent_dim_n/2:
        for a in axs.flat[int(latent_dim_n/2):]: a.axis('off')

    plt.suptitle(' '.join(filter(None, ['clustering', title_suffix])))
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, '_'.join(filter(None, ['clustering', filename_suffix, '.png']))))
    plt.close(fig)


def set_axes(gg):

    for ax in gg.figure.axes:
        ax.set_xlim((-1,1))
        ax.set_ylim((-1,1))
        ax.tick_params(labelsize=13)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        #plt.setp(ax.spines.values(), linewidth=1)
    for ax in gg.diag_axes:
        ax.set_ylim((0,1.7))
        ax.set_xlim((-1,1))   
        ax.get_yaxis().set_visible(False)
        ax.tick_params(labelsize=13)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        

def plot_clusters_pairplot(latent_coords, cluster_assignments, cluster_centers, filename_suffix=None, fig_dir=None):

    # import ipdb; ipdb.set_trace()
    palette = ['#21A9CE', '#5AD871', '#0052A3', '#008F5F']
    sns.set_style(hep.style.CMS)
    sns.set_style({'axes.linewidth': 0.2})

    N = len(latent_coords)
    feat_names = [r"$z_{"+str(z+1)+"}$" for z in range(latent_coords.shape[1])]
    df = pd.DataFrame(latent_coords,columns=feat_names).append(pd.DataFrame(cluster_centers,columns=feat_names), ignore_index=True)
    df['assign'] = np.append(cluster_assignments, [2, 3]) # add cluster assignemnts + dummy class 2 & 3 for cluster centers

    plot = sns.pairplot(df, hue='assign', plot_kws=dict(alpha=0.75, s=[7]*N+[70]*2), diag_kws=dict(fill=False, lw=1.5), palette=palette)

    set_axes(plot)

    # replace labels
    new_labels = ['assigned 1', 'assigned 2', 'center 1', 'center 2']
    for t, l in zip(plot._legend.texts, new_labels):
        t.set_text(l)

    sns.move_legend(plot, bbox_to_anchor=(0.5,-0.07), loc="lower center", ncol=4, labelspacing=0.8, fontsize=16, title='Cluster', title_fontsize=16, markerscale=1.2)
    plt.tight_layout()
    if fig_dir is not None:
        fig = plot.figure
        fig.savefig(os.path.join(fig_dir,'cluster_assignments_'+filename_suffix+'.png'), bbox_extra_artists=(plot._legend,), bbox_inches="tight")


# not obvious to plot 6D decision boundary in 2D => currently not working!
def plot_svm_decision_boundary(model, latent_coords, cluster_assignemnts, title_suffix=None, filename_suffix=None, fig_dir='.fig'):

    # make regular meshgrid for latent space
    grid_vals = np.linspace(np.min(latent_coords, axis=0), np.max(latent_coords, axis=0), 100)
    mesh_vals = np.meshgrid(*list(grid_vals.T))

    # compute decision boundary in entire latent space
    zz = model.decision_function(np.c_([m.ravel() for m in mesh_vals]))
    zz = zz.reshape(mesh_vals[0].shape) # reshape into latent space dimensionality

    latent_dim_n = latent_coords.shape[1] - 1 if latent_coords.shape[1] % 2 else latent_coords.shape[1] # if num latent dims is odd, slice off last dim
    nrows, ncols = calculate_nrows_ncols(latent_dim_n)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(9, 9))

    for d, ax in zip(range(0, latent_dim_n, 2), axs.flat):
        ax.scatter(latent_coords[:,d], latent_coords[:,d+1], c=cluster_assignemnts, s=1.5, cmap='tab10')
        ax.set_title('dims {} & {}'.format(d+1, d+2), fontsize='small')
        # plot decision boundary
        ax.contour(mesh_vals[d], mesh_vals[d+1], zz, levels=[0], c='black', s=100, alpha=0.5);

    if axs.size > latent_dim_n/2:
        for a in axs.flat[int(latent_dim_n/2):]: a.axis('off')

    plt.suptitle(' '.join(filter(None, ['clustering', title_suffix])))
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, '_'.join(filter(None, ['kmeans_clusters', filename_suffix, '.png']))))
    plt.close(fig)



