import matplotlib.pyplot as plt 
import numpy as np


def main():
    save = True

    train_mse = []
    val_mse = []
    epoch = [i for i in range(100)]

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.axvline(x=0.14, linestyle='--', color='darkred', alpha=0.5)
    plt.plot(time, mt1, marker='.', color='red', label=r'$\rho{}_{igm}=0.15\rho{}_{halo}$')
    plt.plot(np.unique(time), np.poly1d(np.polyfit(time, mt1, 1))(np.unique(time)), color='red', alpha=0.3)
    plt.plot(time, mt2, marker='+', color='blue', label=r'$\rho{}_{igm}=0.015$')
    plt.plot(np.unique(time), np.poly1d(np.polyfit(time, mt2, 1))(np.unique(time)), color='blue', alpha=0.3)
    plt.plot(time, mt3, marker='x', color='green', label=r'$\rho{}_{igm}=0.045$')
    plt.plot(np.unique(time), np.poly1d(np.polyfit(time, mt3, 1))(np.unique(time)), color='green', alpha=0.3)
    plt.xlabel("Time [Gyr]", fontsize=14)
    plt.ylabel(r"$rho_{igm, p}$", fontsize=14)
    plt.legend(loc='upper left')
    plt.tight_layout()

    if save:
        plt.savefig("FIG8.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
