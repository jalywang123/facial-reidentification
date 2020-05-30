import matplotlib.pyplot as plt


def show_plot(iteration, loss, save=False):
    plt.plot(iteration, loss)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    if save:
        plt.savefig("loss_figure.png")
    plt.show()
