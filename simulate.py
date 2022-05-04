import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from bound_utils import simulate
from utils import load_data
from conf import digits


# parameters for simulating acc-rec
ds = 'dxy'
num_sims = 100
classifier = 'svm'
random_state = 123


# load dataset
train_path, test_path = 'data/{}/train_set.json'.format(ds), 'data/{}/test_set.json'.format(ds)
train_samples, test_samples = load_data(train_path), load_data(test_path)

# load trained svm model
path = 'saved/{}/{}.pkl'.format(ds, classifier)

# random generate a list of recall values
recs = np.append([0.0, 1.0], np.random.uniform(low=0.0, high=1.0, size=num_sims))

# compute the accuracy score
acc_scores = simulate(train_samples, test_samples, path, recs)

print('acc. (rec equals to 0): {}'.format(round(acc_scores[0], digits)))
print('acc. (rec equals to 1): {}'.format(round(acc_scores[1], digits)))

# bound
lb = {'dxy': 0.644, 'mz4': 0.646, 'mz10': 0.500}.get(ds)
ub = {'dxy': 0.856, 'mz4': 0.757, 'mz10': 0.706}.get(ds)

fig_path = 'saved/{}/{}-sim.pdf'.format(ds, classifier)
with PdfPages(fig_path) as pdf:
    sns.set_context('paper', font_scale=4.0)
    sns.set_theme(color_codes=True)
    df = pd.DataFrame({'recalls': recs, 'acc_scores': acc_scores})
    ax = sns.regplot(data=df, x='recalls', y='acc_scores')
    ax.axhline(lb, ls='--', c='green')
    ax.axhline(ub, ls='--', c='purple')
    plt.xlabel('Symptom Recall (simulate)')
    plt.ylabel('Disease Accuracy')
    plt.tight_layout()
    pdf.savefig()
    print('saving figures to {}'.format(fig_path))


# results
#           acc.(rec=0) acc.(rec=1)
# dxy       0.635       0.856
# mz4       0.626       0.757
# mz10      0.442       0.706
