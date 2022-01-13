import matplotlib.pyplot as plt
import numpy as np


class IQNVisualization(object):
    """
    Visualization of the output of the DQN, IQN, and EQN agents.

    Args:
        nb_actions (int): Number of possible actions.
        nb_quantiles (int): Number of samples quantiles
        iqn (bool): True for IQN and EQN, estimating the quantile distribution
        cvar_eta (float): Parameter for risk-sensitive training, only considering quantiles up to parameter value
    """
    def __init__(self, nb_actions, nb_quantiles, iqn, cvar_eta=1):
        v_max = 15
        v_min = -15

        self.iqn = iqn
        self.nb_actions = nb_actions
        self.nb_quantiles = nb_quantiles

        # Bars showing Q-values
        self.fig1 = plt.figure()
        self.ax1 = plt.gca()
        self.ax1.title._text = 'Q-values (cruise, go, stop)'
        self.ax1.axis([-1, self.nb_actions, v_min, v_max])
        labels = ('cruise', 'go', 'stop')
        x = np.arange(self.nb_actions)
        self.bars_neutral = self.ax1.bar(x, np.zeros(self.nb_actions), width=0.4)
        self.bars_risk = self.ax1.bar(x + 0.5, np.zeros(self.nb_actions), width=0.4)
        self.text_neutral = [self.ax1.text(i - 0.2, 0, '') for i in range(self.nb_actions)]
        self.text2_neutral = [self.ax1.text(i - 0.2, 5, '') for i in range(self.nb_actions)]
        self.text_risk = [self.ax1.text(i - 0.2 + 0.5, 0, '') for i in range(self.nb_actions)]
        self.text2_risk = [self.ax1.text(i - 0.2 + 0.5, 5, '') for i in range(self.nb_actions)]
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+0+500")

        # quantile function
        self.fig2 = plt.figure()
        self.ax2 = plt.gca()
        self.ax2.title._text = 'Quantile function'
        self.ax2.axis([0, 1, v_min, v_max])
        self.lines = [self.ax2.plot([], [], 'x', c=color, label=labels[ii]) for (ii, color)
                      in zip(range(self.nb_actions), ['y', 'g', 'r'])]
        self.ax2.axvline(cvar_eta, c='k')
        self.ax2.legend()
        mngr2 = plt.get_current_fig_manager()
        mngr2.window.wm_geometry("+700+500")

        # histogram pdf
        z_min = v_min
        z_max = v_max
        p_max = self.nb_quantiles
        self.nb_bins = 100
        self.fig3, (self.ax31, self.ax32) = plt.subplots(2)
        self.ax31.title._text = 'Histogram'
        self.ax31.axis([z_min, z_max, 0, p_max])
        self.hist_neutral = self.ax31.hist(np.ones((self.nb_quantiles, self.nb_actions)) * 1000, bins=self.nb_bins,
                                           range=[z_min, z_max], label=labels, color=['y', 'g', 'r'])
        self.ax32.axis([z_min, z_max, 0, p_max])
        self.hist_risk = self.ax32.hist(np.ones((self.nb_quantiles, self.nb_actions)) * 1000, bins=self.nb_bins,
                                        range=[z_min, z_max], label=labels, color=['y', 'g', 'r'])
        self.ax31.legend()
        self.ax32.legend()
        mngr3 = plt.get_current_fig_manager()
        mngr3.window.wm_geometry("+1400+500")

        self.fig1.canvas.draw()
        self.fig2.canvas.draw()
        self.fig3.canvas.draw()
        self.ax1background = self.fig1.canvas.copy_from_bbox(self.ax1.bbox)
        self.ax2background = self.fig2.canvas.copy_from_bbox(self.ax2.bbox)
        self.ax31background = self.fig3.canvas.copy_from_bbox(self.ax31.bbox)
        self.ax32background = self.fig3.canvas.copy_from_bbox(self.ax32.bbox)
        plt.show(block=False)

    def update_plots(self, action, z_values=None, quantiles=None, z_values_detailed=None, quantiles_detailed=None,
                     q_values=None):
        if not self.iqn:
            q_values = q_values
            for idx, (bar, line) in enumerate(zip(self.bars_neutral, self.lines)):
                bar.set_height(q_values[idx])
                bar.set_color('g')
                self.text_neutral[idx].set_text('%.3f' % q_values[idx])
                self.text2_neutral[idx].set_text('%.3f' % (q_values[idx] - np.max(q_values)))
            if action < self.nb_actions:
                self.bars_neutral[action].set_color('r')
            self.fig1.canvas.restore_region(self.ax1background)
            for (bar, text_, text2_) in zip(self.bars_neutral, self.text_neutral, self.text2_neutral):
                self.ax1.draw_artist(bar)
                self.ax1.draw_artist(text_)
                self.ax1.draw_artist(text2_)
            self.fig1.canvas.blit(self.ax1.bbox)
            self.fig1.canvas.flush_events()
        else:
            q_values_neutral = np.mean(z_values_detailed, axis=0)
            q_values_risk = np.mean(z_values, axis=0)
            for idx, (bar_neutral, bar_risk, line) in enumerate(zip(self.bars_neutral, self.bars_risk, self.lines)):
                bar_neutral.set_height(q_values_neutral[idx])
                bar_neutral.set_color('g')
                bar_risk.set_height(q_values_risk[idx])
                bar_risk.set_color('g')
                self.text_neutral[idx].set_text('%.3f' % q_values_neutral[idx])
                self.text2_neutral[idx].set_text('%.3f' % (q_values_neutral[idx] - np.max(q_values_neutral)))
                self.text_risk[idx].set_text('%.3f' % q_values_risk[idx])
                self.text2_risk[idx].set_text('%.3f' % (q_values_risk[idx] - np.max(q_values_risk)))
                line[0].set_xdata(quantiles_detailed)
                line[0].set_ydata(z_values_detailed[:, idx])
            self.bars_neutral[np.argmax(q_values_neutral)].set_color('r')
            if action < self.nb_actions:
                self.bars_risk[action].set_color('r')

            np_hist_neutral = [np.histogram(z_values_detailed[:, i], self.nb_bins,
                                            range=(self.ax31.axis()[0], self.ax31.axis()[1]))
                               for i in range(self.nb_actions)]
            np_hist_risk = [np.histogram(z_values[:, i], self.nb_bins, range=(self.ax32.axis()[0], self.ax32.axis()[1]))
                            for i in range(self.nb_actions)]

            for i in range(self.nb_actions):
                for j in range(self.nb_bins):
                    self.hist_neutral[-1][i][j].set_height(np_hist_neutral[i][0][j])
                    self.hist_risk[-1][i][j].set_height(np_hist_risk[i][0][j])

            self.fig1.canvas.restore_region(self.ax1background)
            self.fig2.canvas.restore_region(self.ax2background)
            self.fig3.canvas.restore_region(self.ax31background)
            self.fig3.canvas.restore_region(self.ax32background)

            for (bar, text_neutral_, text2_neutral_, text_risk_, text2_risk_) \
                    in zip(self.bars_neutral, self.text_neutral, self.text2_neutral, self.text_risk, self.text2_risk):
                self.ax1.draw_artist(bar)
                self.ax1.draw_artist(text_neutral_)
                self.ax1.draw_artist(text2_neutral_)
                self.ax1.draw_artist(text_risk_)
                self.ax1.draw_artist(text2_risk_)
            self.fig1.canvas.blit(self.ax1.bbox)
            self.fig1.canvas.flush_events()

            for line in self.lines:
                self.ax2.draw_artist(line[0])
            self.fig2.canvas.blit(self.ax2.bbox)
            self.fig2.canvas.flush_events()

            for i in range(self.nb_actions):
                for j in range(self.nb_bins):
                    self.ax31.draw_artist(self.hist_neutral[-1][i][j])
                    self.ax32.draw_artist(self.hist_risk[-1][i][j])
            self.fig3.canvas.blit(self.ax31.bbox)
            self.fig3.canvas.blit(self.ax32.bbox)
            self.fig3.canvas.flush_events()
