import matplotlib
import numpy as np

from performance.core import mews, utils

matplotlib.use('Agg')
from matplotlib import pyplot as plt

path = '/home/alex/mews/images'

mews_scorer = utils.worst_case#utils.max
# mews_scorer.__name__ = 'base'
period = 8
# data_level = False
# mews_code, mews_control, code = mews.load()
# case = mews.create_case(code, mews_code, single_event=False)
# mews.calculate(case, period, mews_scorer, data_level=data_level)
# prepared_case = mews.prepare(case)
# for i, k in enumerate(np.unique(prepared_case[:, 0])):
#     cur_data = case[k]
#     inds = np.where(prepared_case[:, 0] == k)[0]
#     f, ax = plt.subplots(6, 1, figsize=(20, 20))
#     ax[0].stem(prepared_case[inds, 1], prepared_case[inds, 2])
#     ax[0].set_xlim([24, 0])
#     ax[0].set_title('MEWS Score')
#     for j, vital in enumerate(['BLOOD PRESSURE SYSTOLIC', 'PULSE', 'R CPN GLASGOW COMA SCALE SCORE', 'RESPIRATIONS', 'TEMPERATURE']):
#         ax[j+1].stem(-np.array(cur_data['raw_time']), np.array(cur_data['data'][vital]))
#         ax[j+1].set_title(vital)
#         ax[j+1].set_xlim([24, 0])
#
#     ax[j+1].set_xlabel('Time (h)')
#     plt.savefig(f'/home/alex/mews/images/profiles/score_level/{mews_scorer.__name__}{("/" + str(period)) * (period is not False)}/profile_{i}.eps')
#     plt.close()



data_level = True

mews_code, mews_control, code = mews.load()
case = mews.create_case(code, mews_code, False)
mews.calculate_scores(case, period, mews_scorer, data_level=data_level)
prepared_case = mews.prepare(case)

for i, k in enumerate(list(case.keys())):
    cur_data = case[k]
    inds = np.where(prepared_case[:,0] == k)[0]
    cur = prepared_case[inds, :]

    f, ax = plt.subplots(6, 1, figsize=(20, 20))
    ax[0].stem(prepared_case[inds, 1], prepared_case[inds, 2])
    ax[0].set_xlim([24, 0])
    ax[0].set_title('MEWS Score')
    for j, vital in enumerate(['BLOOD PRESSURE SYSTOLIC', 'PULSE', 'R CPN GLASGOW COMA SCALE SCORE', 'RESPIRATIONS', 'TEMPERATURE']):
        ax[j+1].stem(-cur_data['raw_time'], np.array(cur_data['data'][vital]))
        ax[j+1].set_title(vital)
        ax[j+1].set_xlim([24, 0])

    ax[j+1].set_xlabel('Time (h)')
    plt.savefig(f'/home/alex/mews/images/profiles/data_level/{mews_scorer.__name__}{("/" + str(period)) * (period is not False)}/profile_{i}.eps')
    plt.close()