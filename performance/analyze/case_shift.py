from performance.core import mews, metrics
import numpy as np
import matplotlib
matplotlib.use('Agg')

from scipy.stats import wilcoxon

from performance.core import scorers, Process, augmenters

if __name__ == '__main__':
    case = mews.prepare_case_multiple()

    thresh = np.arange(4, 5) + 3

    scorers = [scorers.PosNeg(tmin=0, tmax=np.inf)]
    augmenter = augmenters.ShuffleEventMax(num_samples=1000, max_time=1)

    random_processor = Process(thresholds=thresh, scorers=scorers, augmenter=augmenter).per_data

    augment = augmenters.NoAugment()
    processor = Process(thresholds=thresh, scorers=scorers, augmenter=augmenter).per_data

    counts, raw_counts = metrics.run(case, processor)
    random_counts, random_raw_counts = metrics.run(case, random_processor)
    print(random_counts[0].shape)
    x = ((np.array(random_counts[0])[:, :, 0] - counts[0][:, :, 0]) / case.shape[0]).reshape(-1,)

    print(wilcoxon(x))