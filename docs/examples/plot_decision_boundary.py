"""
==============================================
Visualizing SPN Classifier Decision Boundaries
==============================================

This shows the types of surfaces created when using Sum-Product Networks for
classification tasks.
"""

# Authors: Alexander L. Hayes <hayesall@iu.edu>
# License: MIT

from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.algorithms.MPE import mpe
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.Base import Context
from spn.io.Graphics import plot_spn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

np.random.seed(123)


def plot_decision_function(X, y, clf, ax):
    # Adapted from Guillaume Lemaitre's example (MIT License).
    # https://imbalanced-learn.org/stable/auto_examples/combine/plot_comparison_combine.html
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    grid_data = np.c_[xx.ravel(), yy.ravel(), xx.ravel()]
    grid_data.T[2] = np.nan
    Z = mpe(clf, grid_data).T[2]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor="k")


data = (
    datasets.make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        class_sep=0.8,
        n_clusters_per_class=1,
    ),
    datasets.make_circles(n_samples=1000, factor=0.5, noise=0.05),
    datasets.make_moons(n_samples=1000, noise=0.2),
    datasets.make_blobs(n_samples=1000, random_state=8),
)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
axes = (ax1, ax2, ax3, ax4)

for (X, y), ax in zip(data, axes):

    train_data = np.c_[X, y]

    spn_classification = learn_classifier(
        train_data,
        Context(parametric_types=[Gaussian, Gaussian, Categorical]).add_domains(train_data),
        learn_parametric,
        2,
    )

    plot_decision_function(X, y, spn_classification, ax)
    plt.tight_layout()
