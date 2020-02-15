from performance.core import mews
import numpy as np
import matplotlib
matplotlib.use('Agg')

from collections import defaultdict

from matplotlib import pyplot as plt

plot_path = '/home/alex/mews/images/'

if __name__ == '__main__':
    prepared_case, _ = mews.prepare_case_multiple()
    prepared_control, _ = mews.prepare_control()
    case_times = [] ; case_lengths = []
    control_times = [] ; control_lengths = []
    for data, values in zip([prepared_case, prepared_control], [[case_times, case_lengths],[control_times, control_lengths]]):
        for encounter in np.unique(data[:, 0]):
            cur_data = data[data[:, 0] == encounter, :]
            values[0].append(np.max(cur_data[:, 1]))
            values[1].append(np.sum(cur_data[:, 1] >= 0))
    plt.boxplot([case_lengths, control_lengths])
    plt.title('Boxplot of # MEWS Scores Per Patient')
    plt.xticks([1,2],['Case', 'Control'])
    plt.ylabel('Total # of MEWS Scores')
    plt.savefig(f'{plot_path}data.eps')
    plt.close()

    plt.figure()
    plt.title('Boxplot of Length of Patient Stays')
    plt.boxplot([case_times, control_times])
    plt.xticks([1,2], ['Case', 'Control'])
    plt.ylabel('Length of Patient Stay(h)')
    plt.savefig(f'{plot_path}data_times.eps')

    plt.figure()
    plt.title('Boxplot of MEWS Score Frequency Per Patient')
    plt.boxplot([np.divide(case_lengths, case_times), np.divide(control_lengths, control_times)])
    plt.xticks([1,2], ['Case', 'Control'])
    plt.ylabel('MEWS Frequency (/Hour)')
    plt.savefig(f'{plot_path}data_freqs.eps')

    x = [] ; y = []
    for data in (prepared_case, prepared_control):
        counts = defaultdict(list)
        time = []; times = []
        for patient in np.unique(data[:, 0]):
            inds = np.where(data[:, 0] == patient)[0]
            cur_data = data[inds, :]
            cur_data = cur_data[cur_data[:, 1] >= 0, :]
            for thresh in np.arange(0, 13):
                counts[thresh + 1].append(cur_data[cur_data[:, 2] >= thresh, :].shape[0])
            time.append(cur_data.shape[0])
            times.append(np.max(cur_data[:, 1]))
        x.append(np.divide([np.sum(x) for x in counts.values()], np.sum(time)))
        y.append(np.divide([np.sum(x) for x in counts.values()], np.sum(times)))

    plt.figure()
    plt.plot(np.arange(0, 13), x[0])
    plt.plot(np.arange(0, 13), x[1])
    plt.legend(['Case', 'Control'])
    plt.title('MEWS Warning Characteristic')
    plt.xlabel('MEWS Threshold')
    plt.ylabel('Proportion of Scores Above Threshold')
    plt.savefig(f'{plot_path}warning_proportion.eps')


    plt.figure()
    plt.title('MEWS Warning Rate')
    plt.plot(np.arange(0, 13), y[0])
    plt.plot(np.arange(0, 13), y[1])
    plt.legend(['Case', 'Control'])
    plt.xlabel('MEWS Threshold')
    plt.ylabel('# of Warnings per Hour')
    plt.savefig(f'{plot_path}alarm_rate.eps')
