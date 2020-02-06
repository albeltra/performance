import matplotlib
import numpy as np

from performance.core import mews, utils

matplotlib.use('Agg')
from matplotlib import pyplot as plt

plot_path = '/home/alex/mews/images/'
# Prepared the data
prepared_case = mews.prepare_case_multiple()

minutes = 15
def my_round(x, minutes=minutes):
    return round(x*(60/minutes))/(60/minutes)

prepared_case = prepared_case[prepared_case[:, 1] >= 0]
times = [my_round(x) for x in prepared_case[:, 1]]
prepared_case[:, 1] = times
unique_times = np.unique(times)

avg = []
for i, time in enumerate(unique_times):
    print(i)
    avg.append(utils.find_max_mode(prepared_case[np.where(prepared_case[:, 1] == time)[0], 2].tolist()))

plt.stem(unique_times, avg)
plt.title('MEWS Score')
plt.xlabel('Time (h)')
plt.xlim([24, 0])
plt.savefig(f'{plot_path}profiles_{minutes}.svg', transparent=True)

np.random.seed(0)
mrns = np.random.choice(np.unique(prepared_case[:, 1]), 5)

mews_code, mews_control, code = mews.load()
case = mews.create_case(code, mews_code, False)

for i, k in enumerate(list(case.keys())[:10]):
    cur_data = case[k]
    inds = np.where(prepared_case[:,0] == k)[0]

    f, ax = plt.subplots(6, 1, figsize=(20, 20))
    ax[0].stem(prepared_case[inds, 1], prepared_case[inds, 2])
    ax[0].set_xlim([24, 0])
    ax[0].set_title('MEWS Score')
    for j, vital in enumerate(['BLOOD PRESSURE SYSTOLIC', 'PULSE', 'R CPN GLASGOW COMA SCALE SCORE', 'RESPIRATIONS', 'TEMPERATURE']):
        ax[j+1].stem(-np.array(cur_data['time']), np.array(cur_data['data'][vital]))
        ax[j+1].set_title(vital)
        ax[j+1].set_xlim([24, 0])

    ax[j+1].set_xlabel('Time (h)')
    plt.savefig(f'{plot_path}profile_test_{i}.png', dpi=600)