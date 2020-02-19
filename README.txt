Author: Matan Diamond (mdiamond8@gatech.edu)

Dataset: https://www.kaggle.com/mylesoneill/magic-the-gathering-cards/

Python version: 3.7

Python dependencies and imports: pandas, numpy, json, time, re, collections, sklearn

Scikit learn: https://scikit-learn.org/0.15/index.html

# Project organization:
- DataCleaner.py: Requires 'AllCards.json' and 'AllPrintings.json' in 'data' directory.
		  Outputs cleaned dataset to 'data/CleanedDataset.json'
- MtgColor.py:    Requires 'CleanedDataset.json' and 'Keywords.json' in 'data' directory.
		  Outputs a number of files to 'Results' directory including:
			* performance results for each of the three algorithms
			* time results for each of the three algorithms
- Graphs.py:	  Requires a populated 'Results' directory from running MtgColor
		  Outputs algorithm_comparison.png and time_comparison.png to 'Results'
		  	  prints metrics to console for each algorithm.

# Run directions:
To reproduce the data and graphs, simply run DataCleaner, MtgColor and Graphs in that order.

To rerun any part you can run each file given the requirements are present.

To test the entire project, clear the data and Results folders, keeping only the requirements for
Datacleaner and run the files in the order described above. Make sure the folders still exist, even
if empty.

> In terminal, with all dependencies:
python DataCleaner.py
python MtgColor.py
python Graphs.py