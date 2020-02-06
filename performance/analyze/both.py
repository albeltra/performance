import matplotlib
import numpy as np

from performance.core import mews, metrics

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from performance.core import scorers, Process, utils
from performance.core import augmenters

mews_scorer = utils.base
period = False
data_level = False

plot_path = '/home/alex/mews/images/metrics/'

if data_level:
    plot_path += 'data_level/'
else:
    plot_path += 'score_level/'

case = mews.prepare_case_multiple(period=period, scorer=mews_scorer, data_level=data_level)
control = mews.prepare_control(period=period, scorer=mews_scorer, data_level=data_level)

thresh = np.arange(0, 14)
case_scorers = [scorers.PosNeg(tmin=0, tmax=np.inf)]

augmenter = augmenters.NoAugment()

case_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data

case_count, case_count_raw = metrics.run(case, case_processor)

control_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data
control_count, control_count_raw = metrics.run(control, control_processor)

for cur_case, cur_control, scorer in zip(case_count, control_count, case_scorers):
    if scorer.__class__.__name__ == 'PosNeg':
        cur_case = np.transpose(cur_case, (2, 1, 0))
        cur_control = np.transpose(cur_control, (2, 1, 0))

        TP = cur_case[0].reshape(-1,);  FN = cur_case[1].reshape(-1,)
        FP = cur_control[0].reshape(-1,);  TN = cur_control[1].reshape(-1,)

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

        print('TPR, FPR, TNR, FNR, NPV, PPV, WDR', 'ACC', 'F1')
        print([np.round(x[4], 3) for x in [TPR, FPR, TNR, FNR, NPV, PPV, WDR, ACC, F1]])
        print([np.round(x[5], 3) for x in [TPR, FPR, TNR, FNR, NPV, PPV, WDR, ACC, F1]])

        plt.figure()
        plt.plot(FPR, TPR)
        plt.plot(np.arange(0, 1.1, .1), np.arange(0, 1.1, .1), 'k--')
        inds = np.argsort(FPR)
        auc = np.trapz(TPR[inds], FPR[inds])
        plt.legend(['AUC = %.3f' % auc])
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(f'{plot_path}{mews_scorer.__name__}/roc_{period}.eps')
        plt.close()

        plt.figure()
        real_inds = ~np.isnan(PPV)
        plt.plot(TPR, PPV)
        plt.plot(np.arange(0, 1.1, .1), [total_positives / (total_positives + total_negatives)] * 11, 'k--')
        plt.title('Precision v Recall Curve')
        plt.legend(['AUC = %.3f' % np.trapz(PPV[real_inds][::-1], TPR[real_inds][::-1])])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(f'{plot_path}{mews_scorer.__name__}/precision_recall_{period}.eps')
        plt.close()