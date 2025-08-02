import numpy as np
import matplotlib.pyplot as plt

DEFAULT_FIGURE_SIZE = (10, 8)

def plot_bode_diagram(frequencies: np.typing.ArrayLike, response_in_frequency: np.typing.ArrayLike, path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=DEFAULT_FIGURE_SIZE)
    x = frequencies
    y1 = 20 * np.log10(np.abs(response_in_frequency))
    y2 = np.angle(response_in_frequency)*180./np.pi
    ax1.plot(x, y1)
    ax2.plot(x, y2)
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax1.set_title('Bode\'s Diagram: Magnitude')
    ax2.set_title('Bode\'s Diagram: Phase')
    plt.savefig(path)
    return