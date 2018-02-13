#!/usr/bin/env python3

"""
Calculating distances as in Parnas et al. 2013, to make sure everything is
correct, and we can calculate the distances the same way for other odors.
"""

from drosolf import pns
import pandas as pd
import numpy as np
import os

datafile_name = 'table_data.csv'
if not os.exists(datafile_name):
    from . import extract_tables
    table_data = extract_tables.

pn_responses = pns.pns()

figures_we_care_about = {'sx', ...}

for f in figures:
    # get all pairs of odors mentioned in this figure
    # TODO order shouldnt matter, right?
    odor_pairs = set(table_data[f][0]) & set(table_data[f][1])

    # calculate Euclidean and cosine distances between for these odors pairs
    pns_to_odors = pn_responses[
    distances = np.distance(pn_responses)
    angles = np.cosine(pn_responses)

    # sort as in supplement
    distances = np.argsort(distances)
    angles = np.argsort(angles, distances)

    for i, _ in enumerate(odor_pairs):
            
