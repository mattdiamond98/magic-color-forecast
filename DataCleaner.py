import pandas as pd
import numpy as np
import datetime as dt

# After some deliberation, I have only included creature cards to improve performance overall.
#
# Excluded some cards for the following reasons:
# - UNH, UGL, UST, MYSTPT are non-standard cards ("joke" sets)
# - TSP, PLC, FUT are "outlier" blocks that do not adhere to the color pie
# - Cards with a printing before July 21, 2003 excluded because of a major shift in color philosophy detailed here:
# - https://magic.wizards.com/en/articles/archive/making-magic/small-change-2003-07-21
# - Cards with more than one side because they complicate the data unnecessarily
EXCLUDE_SETS_BEFORE = dt.datetime(2003, 7, 21)
EXCLUDED_SET_CODES = {'UNH', 'UGL', 'UST', 'MYSTPT', 'TSP', 'PLC', 'FUT'}
INCLUDED_FIELDS = ['colors', 'convertedManaCost', 'power', 'toughness', 'types', 'subtypes', 'text']

if __name__ == "__main__":
	dataset = pd.read_json('data/AllCards.json')

	print(f"read in dataset of size {dataset.shape}")

	all_printings = pd.read_json('data/AllPrintings.json', convert_dates=True)
	early_sets = list(all_printings.iloc[:,
	                  np.where(all_printings.loc['releaseDate'].astype('datetime64[ns]') < EXCLUDE_SETS_BEFORE)[0]])
	EXCLUDED_SET_CODES.update(early_sets)
	dataset = dataset.iloc[:, np.where(dataset.loc['types'].apply(lambda x: 'Creature' in x))[0]]
	dataset = dataset.iloc[:, np.where(dataset.loc['printings'].apply(lambda x: EXCLUDED_SET_CODES.isdisjoint(x)))[0]]
	dataset = dataset.iloc[:, np.where(dataset.loc['side'] != dataset.loc['side'])[0]]
	dataset = dataset.loc[INCLUDED_FIELDS]

	print(f"writing cleaned dataset of size {dataset.shape}")

	json_dataset = dataset.to_json('data/CleanedDataset.json')

	print(f"data cleaning complete")