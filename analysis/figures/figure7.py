import matplotlib.pyplot as plt 
import numpy as np


def main():
    save = True

    mt1 = [4.075839, 3.2498116, 3.4604836, 7.6339912, 8.771959, 8.53867, 9.801298, 11.628788]
    mt2 = [2.2217317, 1.9898262, 0.8331039, 0.4879744, 1.0081785, 1.0644624, 2.0807796, 3.057458]
    mt3 = [3.66841, 2.3867025, 3.3109665, 1.4311175, 3.7455359, 2.6881871, 3.304738, 3.4424925]
    time = [0.035 * i for i in range(1, 9)]

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
