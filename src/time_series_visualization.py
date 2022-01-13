"""
This code is used to display how any state is developing during an episode. Currently, the class is used
by run_trained_agent.py to display the speed and the epistemic uncertainty as a function of time.
"""
import matplotlib.pyplot as plt
import numpy as np


class TimeSeriesVisualization(object):
    def __init__(self, nb_plots, nb_lines_per_plot, y_range, x_range=(0, 100), labels=None, titles=None):
        self.x_range = x_range
        self.y_range = y_range
        self.fig = []
        self.values = []
        for n in nb_lines_per_plot:
            self.values.append(np.full([n, self.x_range[1] + 1], np.nan))
        self.ax = []
        self.axbackground = []
        self.lines = []
        color = ['b', 'y', 'g', 'r', 'k'*10]

        for i in range(nb_plots):
            self.fig.append(plt.figure())
            self.ax.append(plt.gca())
            self.ax[i].axis([self.x_range[0], self.x_range[1], self.y_range[i][0], self.y_range[i][1]])
            lines = []
            for j in range(nb_lines_per_plot[i]):
                label = labels[i][j] if labels else None
                lines.append(self.ax[i].plot(range(int(self.ax[i].axis()[1]+1)), self.values[i][j, :],
                                             label=label, c=color[j]))
            self.lines.append(lines)
            self.ax[i].legend()
            if titles:
                self.ax[i].title._text = titles[i]
            mngr = plt.get_current_fig_manager()
            mngr.window.wm_geometry("+"+str(500+i*600)+"+00")
            self.fig[i].canvas.draw()
            self.axbackground.append(self.fig[i].canvas.copy_from_bbox(self.ax[i].bbox))
            plt.show(block=False)

    def clear_plots(self):
        for i in range(len(self.values)):
            self.values[i] *= np.nan

    def update_plots(self, new_values):
        for i, values in enumerate(new_values):
            add_idx = np.argwhere(np.isnan(self.values[i]))[0][1]
            self.values[i][:, add_idx] = values
        for i, lines in enumerate(self.lines):
            self.fig[i].canvas.restore_region(self.axbackground[i])
            for j, line in enumerate(lines):
                line[0].set_ydata(self.values[i][j, :])
                self.ax[i].draw_artist(line[0])
                self.fig[i].canvas.blit(self.ax[i].bbox)
                self.fig[i].canvas.flush_events()
