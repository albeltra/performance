import matplotlib
import numpy as np

from performance.core import mews, metrics

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from performance.core import scorers, Process
from performance.core import augmenters

plot_path = '/home/alex/mews/images/'

case, _ = mews.prepare_case_multiple()
control, _ = mews.prepare_control()

thresh = np.arange(4, 5)
case_scorers = [scorers.PosNeg(tmin=0, tmax=np.inf)]
augmenter = augmenters.NoAugment()
case_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data

case_count, case_count_raw = metrics.run(case, case_processor)

augmenter = augmenters.RandomWindows(num_samples=10000)
control_processor = Process(thresholds=thresh, scorers=case_scorers, augmenter=augmenter).per_data
control_count, control_count_raw = metrics.run(control, control_processor)

for cur_case, cur_control, scorer in zip(case_count, control_count, case_scorers):
    if scorer.__name__ == 'PosNeg':
        TP = cur_case[:,:, 0];  FN = cur_case[:, :, 1]
        FP = cur_control[:, :, 0] ;  TN = cur_control[:, :, 1]
        print(FP)

        total_positives = len(np.unique(case[:, 0]))
        total_negatives = len(np.unique(control[:, 0]))
        print(total_negatives)

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
        FPR = FP / (total_negatives)
        # False negative rate
        FNR = FN / (total_positives)

        print('TPR', 'FPR', 'PPV', 'NPV', 'TNR', 'FNR', 'WDR')
        print('Mean')
        print(np.array([np.mean(x) for x in [TPR, FPR, PPV, NPV, TNR, FNR, WDR]]).T)
        print('Std')
        print(np.array([np.std(x) for x in [TPR, FPR, PPV, NPV, TNR, FNR, WDR]]).T)

        plt.figure()
        # bins = np.arange(0,1.001,.001)
        # for x in [FPR, PPV, NPV, TNR, WDR]:
        #     plt.hist(x, bins=bins, density=False, weights=np.ones(len(x))/len(x))

        plt.xlabel('Performance Metric Distributions w/ MEWS >= 4')
        plt.boxplot([FPR.reshape(-1,), PPV.reshape(-1,), NPV.reshape(-1,), TNR.reshape(-1,)])
        plt.xticks(np.arange(0,5), ['FPR', 'PPV', 'NPV', 'TNR', 'WDR'])
        plt.savefig(f'{plot_path}randomized_metric_distributions.svg', transparent=True)