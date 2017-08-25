import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np


class _CallablePlotter:
    """Private class used as the process callback to circumvent the fact that methods are not serializable"""
    def __init__(self):
        self.xdata, self.ydata = [], []
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.draw()
        self.line, = plt.plot(self.xdata, self.ydata)

    def __call__(self, pipe):
        while True:
            if pipe.poll():
                data = pipe.recv()
                self.xdata.append(data[0])
                self.ydata.append(data[1])
                self.line.set_data(np.array(self.xdata), np.array(self.ydata))
                self.ax.relim()
                self.ax.autoscale_view()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            # Allow the UI to update
            plt.pause(0.01)


class Plotter:
    """
    A utility class for live plotting in a separate UI thread, preventing the plot window from being unresponsive in
    between updates.

    The new thread and plot window is spawned upon initialization. Simply call :func:`plot(x, y)` on an instance of this
    class and the point (x, y) will be added to the corresponding plot window without blocking.

    .. note::
        If the main thread is terminated with a KeyboardInterrupt (Ctrl+C), the plot window will stay open, allowing
        one to explore and save the plot if desired.
    """
    def __init__(self):
        self.receive, self.send = mp.Pipe(duplex=False)
        # Separate UI thread
        self.process = mp.Process(target=_CallablePlotter(), args=(self.receive,))
        self.process.start()

    def plot(self, x, y):
        """Adds the point (x, y) to the plot window without blocking"""
        self.send.send((x, y))