from function import Function
import matplotlib.pyplot as plt

class Comparison:
    def __init__(self):
        self.functions: list[Function] = []

    def add_function(self, function):
        self.functions.append(function)

    def plot_all(self, to_one=False):
        ncols = len(self.functions)
        if to_one:
            ncols = 1

        fig, axes = plt.subplots(len(self.functions), ncols, figsize=(15, 10))
        for i, func in enumerate(self.functions):
            if ncols > 1:
                axes[i, 0].plot(func.time_set, func.data_set)
                axes[i, 0].set_title(f"Function {func.name}")
                axes[i, 0].set_xlabel("Time")
                axes[i, 0].set_ylabel("Amplitude")
                axes[i, 1].plot(func.fft_freq, func.fft_set)
                axes[i, 1].set_title(f"FFT {func.name}")
                axes[i, 0].set_xlabel("Frequency")
                axes[i, 0].set_ylabel("Amplitude")
            else:
                axes[0].plot(func.time_set, func.data_set)
                axes[0].set_title(f"Function {func.name}")
                axes[0].set_xlabel("Time")
                axes[0].set_ylabel("Amplitude")
                axes[1].plot(func.fft_freq, func.fft_set)
                axes[1].set_title(f"FFT {func.name}")
                axes[1].set_xlabel("Frequency")
                axes[1].set_ylabel("Amplitude")
        plt.show()


