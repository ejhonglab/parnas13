#!/usr/bin/env python3

"""
Calculating distances as in Parnas et al. 2013, to make sure everything is
correct, and we can calculate the distances the same way for other odors.
"""

from os.path import exists
import pickle

from drosolf import pns

import pandas as pd
import numpy as np

pdf = 'parnas13_supplement.pdf'
# TODO way to automatically get name of figure at same time? -> save to csv
# <fig_name>.csv
pages = [13, 15, 16, 17, 18, 20]
fig2page = {'s{}'.format(i + 1): p for i, p in enumerate(sorted(pages))}
fig2df = {k: None for k in fig2page.keys()}
regenerate_csvs = True

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

        # format table to have correct column names
        # TODO will this work if there is only (either) an "Unnamed: 2" or
        # "Odors"? (trying to rename both to 'Odor B') what about if both are
        # there?
        df.rename(columns={'Unnamed: 0': 'number', 'Unnamed: 1': 'Odor A', \
            'Unnamed: 2': 'Odor B', 'Odors': 'Odor B', 'Decision bias (%)': \
            'Decision bias'}, inplace=True)

        # TODO get rid of spaces?
        df.rename(columns=lambda s: s.lower(), inplace=True)

        print(df)
        print('')
        df.drop(0, inplace=True)

        df['decision bias SEM'] = df['decision bias'].split(u' ± ')

        if f == 's1':
            df.drop(0, inplace=True)

        '''
        elif f == 's2':
        elif f == 's3':
        elif f == 's4':
        elif f == 's5':
        elif f == 's6':
        '''
        # TODO need to drop that extra row of labels in S3? other reformatting?

        # normalize odor names if necessary (for interoperability w/ drosolf or
        # DoOR)

        # encode odor pairs as frozensets? can i put those in dfs?

        fig2df[f] = df
        df.to_csv(datafile)

pn_responses = pns.pns()
'''
n_odors_hallem = 
n_glomeruli_hallem = 

# get all pairs of odors mentioned in this figure
# TODO order shouldnt matter, right?
odor_pairs = set(table_data[f][0]) | set(table_data[f][1])

# calculate Euclidean and cosine distances between for these odors pairs
distance_matrix = np.zeros(
distances = np.distance(pn_responses)
angles = np.cosine(pn_responses)

# sort as in supplement
distances = np.argsort(distances)
angles = np.argsort(angles, distances)

for i, _ in enumerate(odor_pairs):
'''

# TODO how different are 1-OCT and 3-OCT? 3-OCT and MCH?
