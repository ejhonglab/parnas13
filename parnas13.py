#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Calculating distances as in Parnas et al. 2013, to make sure everything is
correct, and we can calculate the distances the same way for other odors.
"""

from os.path import exists
import pickle

import pandas as pd
import numpy as np
from scipy import spatial
import ipdb
try:
    import rpy2.robjects.packages as rpackages
    have_rpy2 = True
except ImportError:
    have_rpy2 = False

from drosolf import pns

'''
pdf = 'parnas13_supplement.pdf'
# TODO way to automatically get name of figure at same time? -> save to csv
# <fig_name>.csv
pages = [13, 15, 16, 17, 18, 20]
fig2page = {'s{}'.format(i + 1): p for i, p in enumerate(sorted(pages))}
fig2df = {k: None for k in fig2page.keys()}
regenerate_csvs = True

# will be extra work to parse these. may try it later.
# TODO why is s5 parsed so badly? different tabula settings?
skip = {'s3', 's5'}


def split_joined_column(series):
    splt = series.str.split(' ')
    left_series = splt.apply(lambda s: ''.join(s[:len(s) // 2]))
    right_series = splt.apply(lambda s: ''.join(s[len(s) // 2:]))
    return left_series, right_series


def make_tidy(df):
    """
    Args:
        df: a DataFrame with columns ['number', 'odor a', 'odor b', 
            'euclidean distance', 'cosine distance', ...]
            where ... is many columns, each with a junk column name,
            and each containing the genotype, training temperature, and testing
            temperature, respectively, in the first three rows. Each subsequent
            row should have a string with the decision bias mean +/- decision
            bias SEM for the flies described by the first three rows.

    Returns a dataframe where there is only one column for decision bias mean
    and one column for decision bias SEM. There will be three extra columns
    to describe the genotype and training and testing temperatures.

    The number, odors, and distances will be copied such that each decision bias
    will still be paired to the same of each of these.
    """
    # much thanks to tdsmith on the #pydata irc channel for a starting point for
    # this code
    label_columns = ['number', 'odor a', 'odor b', 'euclidean distance', \
        'cosine distance']
    labels = df.loc[3:, label_columns].reset_index()

    # TODO only add train / test (temperature) columns if won't be all nan?
    # something similar w/ genotype? do all others actually just parse to CS?
    # TODO be careful not to drop data rows in cases where there aren't
    # temperature or whatever extra metadata rows

    # TODO same as [3:,:]?
    values = df.loc[3:, ].drop(label_columns, axis=1).T
    values["genotype"] = df.iloc[0,:].drop(label_columns)
    values["train"] = df.iloc[1,:].drop(label_columns)
    values["test"] = df.iloc[2,:].drop(label_columns)

    tidy = values.melt(id_vars=["genotype", "train", "test"], var_name="index")

    merged = pd.merge(labels, tidy)

    merged['value'] = merged['value'].apply(lambda s: ''.join(s.split()))
    mean_and_sem = merged['value'].str.split(u'±')
    merged['decision bias mean'] = pd.to_numeric(mean_and_sem.apply( \
        lambda x: x[0]))
    merged['decision bias SEM'] = pd.to_numeric(mean_and_sem.apply( \
        lambda x: x[1]))
    merged.drop('value', axis=1, inplace=True)

    def normalize_temp(s):
        if s is np.nan:
            return np.nan
        elif '25' in s:
            return 25
        elif '32' in s:
            return 32
        else:
            return np.nan
    # TODO special case s2, which has (testing?) temperature specified in
    # different format
    merged['train'] = merged['train'].apply(normalize_temp)
    merged['test'] = merged['test'].apply(normalize_temp)

    def normalize_odor_name(o):
        # DoOR?

        normalized_o = o.lower()

        if normalized_o == 'limonen':
            normalized_o = 'limonene'

        elif normalized_o == 'δ-decalactone':
            normalized_o = 'd-decalactone'

        # TODO is hallem spontaneous rate the mineral oil reponse?
        # it might actually be kind of important...
        elif normalized_o == 'mineral oil':
            normalized_o = 'spontaneous firing rate'

        return normalized_o

    merged['odor a'] = merged['odor a'].apply(normalize_odor_name)
    merged['odor b'] = merged['odor b'].apply(normalize_odor_name)

    return merged


for f, p in fig2page.items():
    datafile = 'parnas_{}.csv'.format(f)

    if not regenerate_csvs and exists(datafile):
        fig2df[f] = pd.read_csv(datafile)

    else:
        # how does the package tabula differ from tabula-py? 
        # (both imported this way)
        import tabula

        # TODO pull request tabula-py readme change to change obsolete import to
        # read_pdf rather than read_pdf_table

        # TODO consider reading remote PDFs, to avoid copyright issues
        # read_pdf(url)

        #dfs = tabula.read_pdf(pdf, pages=pages, multiple_tables=True)
        df = tabula.read_pdf(pdf, pages=p)
        if df is None:
            print('Warning: no table could be parsed from ' + \
                'requested page {}'.format(p))
            continue

        #if f != 's4':
        #    continue

        if f in skip:
            print('Skipping {}'.format(f))
            continue

        # drop any columns only containing NaN
        df.dropna(axis=1, how='all', inplace=True)

        # format table to have correct column names
        # TODO will this work if there is only (either) an "Unnamed: 2" or
        # "Odors"? (trying to rename both to 'Odor B') what about if both are
        # there?
        df.rename(columns={'Unnamed: 0': 'number', 'Unnamed: 1': 'Odor A', \
            'Unnamed: 2': 'Odor B', 'Odors': 'Odor B', 'Decision bias (%)': \
            'Decision bias'}, inplace=True)

        # TODO get rid of spaces?
        df.rename(columns=lambda s: s.lower(), inplace=True)

        if f == 's2':
            # TODO rename second to last col to decision bias ('unnamed: 6')
            # and add a column for genotype, filling in [0, -1] for all [:, -1]
            # wt for all [:, -2]
            # delete current col for decision bias, or delete -2
            df.rename(columns={'unnamed: 3': 'odor b'}, inplace=True)

        elif f == 's3':
            pass

        elif f == 's4':
            # ['unnamed: 7']['untrained at 320C'] -> ['trained_at'] =
            # 'not_trained', ['tested_at'] = 32, ['decision bias {mean/SEM}'] =
            # ['unnamed: 7'][3:]
            # drop 0-2
            # TODO refactor into function to fix joined column?
            tested25, tested32 = split_joined_column(df['decision bias'])
            df['tmp1'] = tested25
            df['tmp2'] = tested32
            df.drop('decision bias', axis='columns', inplace=True)

        elif f == 's5':
            pass

        elif f == 's6':
            # TODO separate distances into different columns by genotype 
            # in row 0, or add another column with that genotype
            # TODO drop cols unnamed: 6&7
            # TODO get col which has two sets of decision biases lumped together
            # separated
            pass
        #print('************* START **************')
        print(f)
        #print(df.columns)
        #print(df)
        df = make_tidy(df)
        print(df.columns)
        print(df)
        #print('************* END **************')
        print('')

        fig2df[f] = df
        df.to_csv(datafile)

odors = set()
odor_pairs = set()
pair2euclidean = dict()
pair2cosine = dict()

for df in fig2df.values():
    if df is None:
        continue

    for _, row in df.iterrows():
        odors.add(row['odor a'])
        odors.add(row['odor b'])
        fs = frozenset((row['odor a'], row['odor b']))
        odor_pairs.add(fs)
        pair2euclidean[fs] = row['euclidean distance']
        pair2cosine[fs] = row['cosine distance']

pn_responses = pns.pns()

# TODO is the index supposed to be 111 long? more?
hallem_odors = set(pn_responses.index)
assert len(odors - hallem_odors) == 0, 'no ORN data for some odors!'

for fs in odor_pairs:
    o1, o2 = fs
    pn1 = pn_responses.loc[o1]
    pn2 = pn_responses.loc[o2]

    recalculated_euclidean = np.linalg.norm(pn1 - pn2)
    # TODO also calculate with my own function
    recalculated_cosine = spatial.distance.cosine(pn1, pn2)
    # TODO way to print table like things in python? this looks ugly as hell
    print('       Theirs | Ours')
    print('{}, {}:   {}    {}'.format(o1, o2, pair2euclidean[fs], \
        recalculated_euclidean))
    print('                 cosine    {}    {}'.format(pair2cosine[fs], \
        recalculated_cosine))
'''

# TODO how different are 1-OCT and 3-OCT? 3-OCT and MCH?
if have_rpy2:
    from rpy2.robjects.vectors import StrVector
    from rpy2.robjects.packages import importr
    from rpy2 import robjects as ro

    # TODO how to check if r package is already installed?
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)

    if not rpackages.isinstalled('devtools'):
        # TODO way to install package as personal library + create one
        # by default? or run just this command as root?
        print(dir(utils))
        utils.install_packages(StrVector(('devtools',)))
        print(utils.installed_packages())

    else:
        print('devtools installed!')

    devtools = importr('devtools')

    # do i need to use ro.r(...) ever? or can i do all without that? 
    # TODO make timeout longer / retry?
    devtools.install_github('ropensci/DoOR.data')
    #devtools.install_github('ropensci/DoOR.functions')

    door_data = importr('DoOR.data')
    print(dir(door_data))

    # TODO what is the the nointeraction=TRUE bit for, in load_door_data(...)?
    # TODO way to load it to a dataframe? or otherwise how to get it out of
    # the r workspace / inspect the workspace to see what it loaded?
    door_data.load_door_data()
    

