import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np


class _CallablePlotter:
    """Private class used as the process callback to circumvent the fact that methods are not serializable"""
    def __init__(self, subplots, titles):
        self.lines = [[] for _ in subplots]
        self.xdata, self.ydata = [[[] for _ in range(m)] for m in subplots], [[[] for _ in range(m)] for m in subplots]
        self.fig, self.axes = plt.subplots(len(subplots), sharex=True)
        self.fig.canvas.draw()
        for i in range(len(subplots)):
            self.axes[i].set_title(titles[i])
            for _ in range(subplots[i]):
                self.lines[i].append(self.axes[i].plot([], [])[0])

    def __call__(self, pipe):
        while True:
            if pipe.poll():
                x, y, n, m = pipe.recv()
                self.xdata[n][m].append(x)
                self.ydata[n][m].append(y)
                self.lines[n][m].set_data(np.array(self.xdata[n][m]), np.array(self.ydata[n][m]))
                for ax in self.axes:
                    ax.relim()
                    ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            # Allow the UI to update
            plt.pause(0.01)


class Plotter:
    """
    A utility class for live plotting in a separate UI thread, preventing the plot window from being unresponsive in
    between updates.

    The new thread and plot window is spawned upon initialization. Simply call :func:`plot(x, y, n)` on an instance of
    this class and the point (x, y) will be added to the nth subplot without blocking.

    .. note::
        If the main thread is terminated with a KeyboardInterrupt (Ctrl+C), the plot window will stay open, allowing
        one to explore and save the plot if desired.
    """
    def __init__(self, subplots, *titles):
        """
        :param subplots: list of integers, the length of which will be the number of subplots, with each integer
        representing the number of lines in the corresponding subplot
        :param titles: list of titles for the subplots, must be same length as subplots
        """
        self.receive, self.send = mp.Pipe(duplex=False)
        # Separate UI thread
        self.process = mp.Process(target=_CallablePlotter(subplots, titles), args=(self.receive,))
        self.process.start()

    def plot(self, x, y, n, m):
        """Adds the point (x, y) to the mth line on the nth subplot without blocking"""
        self.send.send((x, y, n, m))