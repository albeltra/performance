import matplotlib
import numpy as np

from performance.core import mews, utils

matplotlib.use('Agg')
from matplotlib import pyplot as plt

path = '/home/alex/mews/images/profiles/'


for data_level in [True, False]:
    l = [np.max, utils.mode, np.median, np.mean]
    if data_level:
        l += [utils.worst_case]
    for mews_scorer in l:
        print(mews_scorer.__name__)
        for period in [1, 2, 4, 8]:
            prepared_case, case = mews.prepare_case_multiple(period=period, scorer=mews_scorer, data_level=data_level)

            for i, k in enumerate(np.unique(prepared_case[:, 0])):
                cur_data = case[k]
                inds = np.where(prepared_case[:, 0] == k)[0]
                f, ax = plt.subplots(6, 1, figsize=(20, 20))
                ax[0].stem(prepared_case[inds, 1], prepared_case[inds, 2])
                ax[0].set_xlim([24, 0])
                ax[0].set_title('MEWS Score')
                ax[1].set_title('Vital Signs for Calculated MEWS')
                for j, (vital, label) in enumerate(zip(['BLOOD PRESSURE SYSTOLIC', 'PULSE', 'R CPN GLASGOW COMA SCALE SCORE', 'RESPIRATIONS', 'TEMPERATURE'],
                                                             ['Systolic Blood Pressure (mmHg)', 'Heart Rate (BPM)', 'Glascow Coma Scale Score', 'Respiration Rate (BPM)', 'Temperature ($^\circ$F)'])):
                    ax[j + 1].stem(-np.array(cur_data['raw_time']), np.array(cur_data['data'][vital]))
                    ax[j + 1].set_xlim([24, 0])
                    ax[j + 1].set_ylabel(label)

                ax[j + 1].set_xlabel('Time (h)')
                if data_level:
                    plt.savefig(
                        f'{path}data_level/{mews_scorer.__name__}{("/" + str(period)) * (period is not False)}/profile_{i}.eps')
                else:
                    plt.savefig(
                        f'{path}score_level/{mews_scorer.__name__}{("/" + str(period)) * (period is not False)}/profile_{i}.eps')
                plt.close()