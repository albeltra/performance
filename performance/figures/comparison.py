import numpy as np
from matplotlib import pyplot as plt

from performance.core import augmenters
from performance.core import mews, metrics
from performance.core import scorers, Process, utils

thresh = np.arange(0, 14)
lead_times = np.arange(0, 4, 10 / 60)
case_scorers = [scorers.PosNeg(tmin=0, tmax=np.inf),
                scorers.Lead(lead_times=lead_times)]
augmenter = augmenters.NoAugment()
periods = [2,4,8]


fig, ax = plt.subplots(3, 2, figsize=(10 ,10), sharey=True)

for p, a in zip(periods, ax[:, 0]):
    a.set_ylim([0, 1])
    a.set_ylabel(f'Period = {p} Hours', rotation=90)
for a in ax[:, 1]:
    a.set_ylim([0, 1])


ax[0,0].set_title('ROC Curves')
ax[0, 1].set_title('Sensivity at Lead Time')

ax[-1, 0].set_xlabel('False Positive Rate')
ax[-1, 1].set_xlabel('Lead Time (h)')

plot_path = '/home/alex/mews/images/metrics/'

for f, period in enumerate(periods):
    for i, (mews_scorer, data_level) in enumerate(zip([utils.base, np.median, utils.worst_case], [False, False, True])):
        if mews_scorer.__name__ == 'base':
            case, _ = mews.prepare_case_multiple(period=False, scorer=mews_scorer, data_level=data_level)
            control, _ = mews.prepare_control(period=False, scorer=mews_scorer, data_level=data_level)
        else:
            case, _ = mews.prepare_case_multiple(period=period, scorer=mews_scorer, data_level=data_level)
            control, _ = mews.prepare_control(period=period, scorer=mews_scorer, data_level=data_level)

        case_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data
        case_count, case_count_raw = metrics.run(case, case_processor)

        control_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data
        control_count, control_count_raw = metrics.run(control, control_processor)

        for s_ind, (cur_case, cur_control, scorer) in enumerate(zip(case_count, control_count, case_scorers)):
            if scorer.__class__.__name__ == 'PosNeg':
                cur_case = np.transpose(cur_case, (2, 1, 0))
                cur_control = np.transpose(cur_control, (2, 1, 0))

                TP = cur_case[0].reshape(-1, )
                FN = cur_case[1].reshape(-1, )
                FP = cur_control[0].reshape(-1, )
                TN = cur_control[1].reshape(-1, )

                total_positives = len(np.unique(case[:, 0]))
                total_negatives = len(np.unique(control[:, 0]))
                # Sensitivity or true negative rate
                TPR = TP / (total_positives)
                # Specificity or true negative rate
                TNR = TN / (total_negatives)
                # Precision or positive predictive value
                PPV = TP / (TP + FP)
                # WDR
                WDR = 1 / PPV
                # Negative predictive value
                NPV = TN / (TN + FN)
                # Fall out or false positive rate
                FPR = FP / total_negatives
                # False negative rate
                FNR = FN / total_positives

                ACC = (TP + TN) / (total_negatives + total_positives)

                F1 = 2 * (PPV * TPR) / (PPV + TPR)

                ax[f, s_ind].plot(FPR, TPR, color='C'+ str(i))
                ax[f, s_ind].legend(['Traditional', 'Median Score', 'Worst Case Vital Sign'], loc='lower right')

        if scorer.__class__.__name__ == 'Lead':
            sensitivity_discount = cur_case[0, :, :, 0] / cur_case[0, :, :, 1]
            ax[f, s_ind].plot(lead_times, sensitivity_discount[4], color='C' + str(i))
            ax[f, s_ind].legend(['Traditional', 'Median Score', 'Worst Case Vital Sign'], loc='lower right')


fig.tight_layout()
fig.show()
fig.savefig(f'/home/alex/mews/images/multiple_periods.png', dpi=600)