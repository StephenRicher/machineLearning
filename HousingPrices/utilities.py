#!/usr/bin/env python3

""" Helper functions for AML data visualisation """

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2
from sklearn.covariance import MinCovDet
from pandas.api.types import is_numeric_dtype


def pearsonr_pval(x,y):
    try:
        return pearsonr(x,y)[1]
    except ValueError:
        return np.nan


def countPair(x, y):
    """ Return count of valid pairs (both not nan) """

    # Indices where both x and y are NOT np.nan
    validIndices = np.intersect1d(
        np.where(~np.isnan(x)),
        np.where(~np.isnan(y)))
    return len(validIndices)


def mahalanobis(data):
    """ Compute Mahalanobis Distance between
        each row of data with the data. """
    x_minus_mu = data - np.mean(data)
    cov = np.cov(data.values.T)
    inv_covmat = linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


def plotNumeric(X, y, out=None, threshold=0.01):
    """ Wrapper to visualise numerical feature distribution """
    if (y.equals(X)):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        df = pd.concat([X], axis=1).dropna()
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        df = pd.concat([X, y], axis=1).dropna()
        df['mahalanobis'] = mahalanobis(df)
        df['outlier'] = df['mahalanobis'] > chi2.ppf((1 - threshold), df=2)
        sns.scatterplot(x=X.name, y=y.name, hue='outlier', data=df, ax=ax3)
        ax3.get_legend().remove()
    sns.distplot(x=df[X.name], ax=ax1)
    ax1.set_title(f'Skewness = {df[X.name].skew():.2f}; '
                  f'Kurtosis = {df[X.name].kurtosis():.2f}')
    stats.probplot(df[X.name], plot=ax2)
    ax2.set_title(f'Probability Plot: {ax1.get_xlabel()}')
    fig.tight_layout()
    if out is not None:
        fig.savefig(out)
    return fig, (ax1, ax2)


def plotCategory(X, y, out=None):
    """ Wrapper to visualise categorical feature distribution """
    df = pd.concat([X, y], axis=1)
    assert 'NAvalues' not in [X.name, y.name]
    df[X.name] = df[X.name].fillna('NAvalues')
    counts = df[X.name].value_counts().sort_values()
    countsSum = df[X.name].value_counts(normalize=True).sort_values().cumsum()
    order = counts.sort_values(ascending=True).index
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    sns.stripplot(x=y.name, y=X.name, alpha=0.5, order=order, data=df, ax=ax1)
    sns.violinplot(x=y.name, y=X.name, order=order, data=df, ax=ax2)
    sns.barplot(counts.values, counts.index, ax=ax3)
    ax3.set_xlabel('Counts')
    sns.barplot(countsSum.values, countsSum.index, ax=ax4)
    ax4.set_xlabel('Cumulative sum proportion of counts')
    fig.tight_layout()
    if out is not None:
        fig.savefig(out)
    return fig, (ax1, ax2)


def plotAllFeatures(df, target, outdir='.'):
    """ Wrapper for plotting functions """
    os.makedirs(outdir, exist_ok=True)
    for col in df.columns:
        out = f'{outdir}/{col}-distribution.png'
        if is_numeric_dtype(df[col]):
            fig, axes = plotNumeric(
                df[col], df[target], out=out)
        else:
            fig, axes = plotCategory(
                df[col], df[target], out=out)


def computeCorrelation(df, p=0.05):
    """ Compute pairwise correlation, p-value and pair counts """
    correlations = []
    for method in ['pearson', pearsonr_pval, countPair]:
        values = df.corr(method=method).stack()
        correlations.append(values)
    correlations = (
        pd.concat(correlations, axis=1)
        .reset_index()
        .rename(columns={'level_0': 'feature1',
                         'level_1': 'feature2',
                         0: 'R', 1: 'p', 2: 'n'}))
    correlations['significant'] = correlations['p'] < p
    return correlations


def plotPairwiseCorrelation(correlations, out=None):
    """ Plot pairwise correlation matrix with
        output from computeCorrelation() """
    wideCorr = correlations.pivot(
        columns='feature1', index='feature2', values='R')
    g = sns.clustermap(
        wideCorr,
        xticklabels=1, yticklabels=1,
        row_cluster=True, col_cluster=True,
        cbar_pos=None, cmap='bwr', vmin=-1, vmax=1)
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.set_xlabel('')
    g.ax_heatmap.set_ylabel('')
    plt.tight_layout()
    if out is not None:
        plt.savefig(out)


def plotTargetCorrelation(correlations, target, out=None):
    """ Plot correlations relative to single target """
    targetCorr = (
        correlations.loc[correlations['feature1'] == target]
        .set_index('feature2'))
    targetCorr = targetCorr.sort_values(by=['R'], ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    targetCorr = targetCorr.loc[targetCorr.index != targetCorr['feature1']]
    sns.heatmap(pd.DataFrame(targetCorr['R']), yticklabels=1,
                cmap='bwr', vmin=-1, vmax=1, ax=ax1)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.tick_params(left=True)
    sns.heatmap(pd.DataFrame(targetCorr['p']), yticklabels=1,
                cmap='viridis', vmin=0, vmax=1, ax=ax2)
    ax2.tick_params(left=True)
    ax2.set_ylabel('')
    fig.tight_layout()
    if out is not None:
        fig.savefig(out)
    return fig, (ax1, ax2)
