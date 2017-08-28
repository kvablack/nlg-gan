import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np


class _CallablePlotter:
    """Private class used as the process callback to circumvent the fact that methods are not serializable"""
    def __init__(self, n, titles):
        self.lines = []
        self.xdata, self.ydata = [[] for _ in range(n)], [[] for _ in range(n)]
        self.fig, self.axes = plt.subplots(n, sharex=True)
        self.fig.canvas.draw()
        for ax, title in zip(self.axes, titles):
            self.lines.append(ax.plot([], [])[0])
            ax.set_title(title)

    def __call__(self, pipe):
        while True:
            if pipe.poll():
                data = pipe.recv()
                self.xdata[data[2]].append(data[0])
                self.ydata[data[2]].append(data[1])
                self.lines[data[2]].set_data(np.array(self.xdata[data[2]]), np.array(self.ydata[data[2]]))
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
    def __init__(self, n, *titles):
        self.receive, self.send = mp.Pipe(duplex=False)
        # Separate UI thread
        self.process = mp.Process(target=_CallablePlotter(n, titles), args=(self.receive,))
        self.process.start()

    def plot(self, x, y, n):
        """Adds the point (x, y) to the nth subplot without blocking"""
        self.send.send((x, y, n))