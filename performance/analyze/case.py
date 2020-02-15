import matplotlib
import numpy as np
from matplotlib.collections import LineCollection

from performance.core import mews

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from performance.core import scorers, Process, metrics
from performance.core import augmenters

mews_scorer = np.max
period = 8
data_level = True

plot_path = '/home/alex/mews/images/metrics/'

if data_level:
    plot_path += 'data_level/'
else:
    plot_path += 'score_level/'

case, _ = mews.prepare_case_multiple(period=period, scorer=mews_scorer, data_level=data_level)

# Set thresholds and scores to be calculated
thresh = np.arange(0, 14)
lead_times = np.arange(0, 6.167, 10/60)
case_scorers = [scorers.PosNeg(tmin=0, tmax=np.inf),
                scorers.Lead(lead_times=lead_times),
                scorers.Profile(tmin=1, tmax=13),
                scorers.ProfileNorm(tmin=1, tmax=13),
                scorers.PerHour(max_time=13, step_size=1)]

augmenter = augmenters.NoAugment()

case_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data

case_count, case_count_raw = metrics.run(case, case_processor)

for cur_case, cur_case_raw, scorer in zip(case_count, case_count_raw, case_scorers):
    if scorer.__class__.__name__ == 'Lead':
        sensitivity_discount = cur_case[0, :, :, 0] / cur_case[0, :, :, 1]
        legend = []
        plt.figure()
        for i, d in enumerate(sensitivity_discount):
            if i==4:
                plt.plot(lead_times, d, 'o')
                legend.append(i)
        plt.legend(legend, title='MEWS (>=)')
        plt.ylim([.4, .9])
        plt.xlabel('Lead Time (h)')
        plt.ylabel('Sensitivity')
        plt.savefig(f'{plot_path}{mews_scorer.__name__}/lead_discount_{period}.eps')
        plt.close()

        sensitivity = cur_case[0, :, :, 0] / len(np.unique(case[:, 0]))
        legend = []
        plt.figure()
        for i, d in enumerate(sensitivity):
            if i==4:
                plt.plot(lead_times, d)
                legend.append(i)
        plt.ylim([.4, 1])
        plt.legend(legend, title='MEWS (>=)')
        plt.xlabel('Lead Time (h)')
        plt.ylabel('Sensitivity')
        plt.savefig(f'{plot_path}{mews_scorer.__name__}/lead_{period}.eps')
        plt.close()

    if scorer.__class__.__name__ == 'PerHour':
        for i, d in enumerate(np.array(cur_case[0])):
            plt.figure()
            print(d.shape)
            sensitivity_discount = d[:, 0] / d[:, 1]
            print(thresh)
            print(sensitivity_discount)
            plt.bar(thresh, sensitivity_discount, align='center')
            plt.ylim([0, 1])
            plt.xticks(np.arange(0, 14))
            plt.xlabel('Hour Before Event')
            plt.ylabel('Number of Warnings Per Event')
            plt.savefig(f'{plot_path}{mews_scorer.__name__}/per_hour_{i}_{period}.eps')
            plt.close()

    if scorer.__class__.__name__ == 'ProfileNorm':
        for i, d in enumerate(np.array(cur_case[0])):
            plt.figure()
            objects = ('Early', 'On Time', 'Late')
            y_pos = np.arange(len(objects))
            d_new = d[:, 0] / d[:, 1]

            plt.figure()
            plt.bar(y_pos, d_new, align='center')
            plt.ylim(0, 1)
            plt.xticks(y_pos, objects)
            plt.ylabel('Proportion of Time')
            plt.title('Warning Distributions')

            plt.savefig(f'{plot_path}{mews_scorer.__name__}/profile_norm_{i}_{period}.eps')
            plt.close()

    if scorer.__class__.__name__ == 'Profile':
        objects = ('Early', 'On Time', 'Late', 'Missed')
        y_pos = np.arange(len(objects))
        for i, d in enumerate(np.array(cur_case[0])):
            new = d.copy()
            new[:-1] /= np.sum(new[:-1])
            new[-1] /= cur_case_raw.shape[0]

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            f = ax1.bar(y_pos, new, align='center')
            f[3].set_color('r')

            ax1.set_ylabel('Proportion of Warnings', color='b')
            ax2.set_ylabel('Proportion Missed Patients', color='r')

            plt.xticks(y_pos, objects)
            plt.title('Warning Distributions')

            plt.savefig(f'{plot_path}{mews_scorer.__name__}/profile_case_{i}_{period}.eps')
            plt.close()

    if scorer.__class__.__name__ == 'Profile':
        objects = ('Early', 'On Time', 'Late', 'Missed')
        y_pos = np.arange(len(objects))
        for i, d in enumerate(np.array(cur_case[0], ndmin=2)):
            new = d.copy()
            new[:-1] /= cur_case_raw.shape[0]
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax2.yaxis.set_ticklabels([''] * len(ax2.get_xticklabels()))
            ax2.yaxis.set_ticks_position('none')
            f = ax1.bar(y_pos, new, align='center')
            f[3].set_color('r')

            ax1.set_ylabel('# of Warnings per Event', color='b')
            ax2.set_ylabel('# Missed Patients', color='r')

            plt.xticks(y_pos, objects)
            plt.title('Warning Distributions')

            plt.savefig(f'{plot_path}{mews_scorer.__name__}/profile_case_warnings_{i}_{period}.eps')
            plt.close()
