import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np


def grid_histogram(data, plot_title, titles, plot_labels, x_labels, y_labels,
                   wspace, hspace, save, filename):
    """
    Plot histogram of data.
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    subtitle_font = {'weight': 'normal', 'size': 12}
    title_font = {'weight': 'bold', 'size': 14}
    text_properties = {'verticalalignment': 'top',
                       'horizontalalignment': 'left'}

    n = len(data)
    dim1 = math.ceil(math.sqrt(n))
    dim2 = math.ceil(n / dim1)
    bins = 40

    fig = plt.figure()
    ax_main = plt.subplot(111)
    ax_main.set_ylabel('Normalised Density')
    for i in range(n):
        dat = data[i]
        ax = plt.subplot(dim2, dim1, i + 1, frame_on=True)
        x, bins, p = ax.hist(dat[dat != 0], bins=bins, density=True)

        ax.tick_params(labelsize=7)

        for item in p:
            item.set_height(item.get_height() / max(x))

        if (i % dim1 == 0):
            ax.set_ylabel(y_labels[int(i / dim1)])
        else:
            ax.set_yticks([])

        if (i < 3):
            ax.set_xticks([])

        if (i < dim1):
            ax.set_title(x_labels[i])

        ax.set_ylim([0.0, 1.1])
        ax.text(0.0, 1.05, titles[i], **text_properties, **subtitle_font)

    fig.text(0.5, 0.04, 'Pixel Value (Normalised)',
             ha='center', va='center', size=12)
    fig.text(0.02, 0.5, 'Frequency (Normalised)', ha='center',
             va='center', rotation='vertical', size=12)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.suptitle(plot_title, **title_font)
    plt.savefig(filename) if save else plt.show()


def grid_images(X, plot_title, titles, x_labels, y_labels,
                wspace, hspace, save, filename):
    """
    Plot a grid of 2D map images.
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    subtitle_font = {'weight': 'normal', 'size': 12, 'color': 'white'}
    title_font = {'weight': 'bold', 'size': 14}

    n = X.shape[0]
    dim1 = math.ceil(math.sqrt(n))
    dim2 = math.ceil(n / dim1)

    assert (len(titles) == n), "Number of titles incorrect (expected %i)" % n

    plt.figure()

    for i in range(n):
        image = np.matrix(X[i])
        ax = plt.subplot(dim2, dim1, i + 1, frame_on=False)
        ax.text(0.1, 0.1, titles[i], verticalalignment='top',
                horizontalalignment='left', **subtitle_font)

        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])

        if (i == 0):
            fontprops = fm.FontProperties(size=8)
            scalebar = AnchoredSizeBar(
                ax.transData, float(20 / 14 * 10), '10 kpc', 'lower right',
                pad=0, color='white', frameon=False, size_vertical=1,
                fontproperties=fontprops)
            ax.add_artist(scalebar)

        if y_labels is not None:
            if (i % dim1 == 0):
                ax.set_ylabel(y_labels[int(i / dim1)])
        if x_labels is not None:
            if (i < dim1):
                ax.set_title(x_labels[i])
        ax.imshow(image)

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.suptitle(plot_title, **title_font, y=1.0)
    plt.savefig(filename) if save else plt.show()


def plot_galaxy(X, y, index):
    """
    Function to plot galaxies from 2D map data
    """
    galaxy_choice = np.matrix(X[index])
    plt.imshow(galaxy_choice, interpolation="nearest")
    plt.title('Galaxy #%i, rho: %.2f' % (index, y[index]))
    plt.show()


def plot_histogram(var, name):
    """
    Simple plot of histogram of a chosen variable
    """
    # plotting histogram of values (for vrel and rho)
    plt.hist(var)
    plt.title("Histogram " + str(name))
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.show()


def plot_compare(pred, true):
    """
    Compare predictions with true values for paper figures
    """
    x_eq = np.arange(0, 10, 1)
    y_eq = np.arange(0, 10, 1)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(true, pred, 'ro', alpha=0.01)
    plt.plot(x_eq, y_eq, color='black', dashes=[2, 2])
    plt.title("2D Density")
    plt.xlabel(r"$y$", fontsize=14)
    plt.ylabel(r"$y_{pred}$", fontsize=14)
    plt.tight_layout()
    plt.show()
