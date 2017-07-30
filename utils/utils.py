import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef,
                             cohen_kappa_score)
from sklearn.model_selection import learning_curve


def summary_statistics(df, y_true, class_names):
    """
    Print an overview of the dataframe.
    
    :param df: a pandas DataFrame
    :param df: a pandas Series with the true labels associated with the DataFrame observations
    :param class_names: a list of the class names
    """
    print('No of transactions: {}'.format(df.shape[0]))
    print('No of features/variables: {}'.format(df.shape[1]))
    print('No of classes: {}'.format(len(class_names)))
    print('\nNo of {} transactions (class = 1): {}'.format(class_names[1], y_true.value_counts()[1]))
    print('No of {} transactions (class = 0): {}'.format(class_names[0], y_true.value_counts()[0]))
    print('\nClass Distribution: {}:1'.format(
        int(round(y_true.value_counts()[0] / y_true.value_counts()[1], ndigits=0))))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_confusion_matrix(cm, class_names,
                          normalise=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """ Print and plot the confusion matrix.
    Normalisation can be applied by setting `normalise=True`. """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def print_report(y_true, y_pred, class_names, normalise=False):
    """
    Display a complete evaluation report for the trained model.
    """
    cnf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    print("ROC AUC: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    print("Matthews Correlation Coefficient: {:0.3f}".format(matthews_corrcoef(y_true, y_pred)))
    print("Cohen's Kappa: {:0.3f}".format(cohen_kappa_score(y_true, y_pred)))
    print("False Positive Rate: {:0.3f}".format(cnf_matrix[0, 1] / (cnf_matrix[0, 1] + cnf_matrix[0, 0])))
    print("False Negative Rate: {:0.3f}".format(cnf_matrix[1, 0] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])))

    # plot the confusion matrix
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, class_names=class_names, normalise=normalise,
                          title='Normalised confusion matrix' if normalise else 'Non-normalised confusion matrix')
    plt.show()

