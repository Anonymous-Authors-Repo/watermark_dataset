import matplotlib.pyplot as plt

def display_waveform(waveform, title="", sr=8000):
    """Display waveform plot and audio play UI."""
    plt.figure()
    plt.plot(waveform, 'k', linewidth=1.0)
    #plt.axis('off')
    plt.tight_layout()
    plt.show()