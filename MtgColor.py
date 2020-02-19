import pandas as pd
import numpy as np
import json
import time
import re
from collections import Counter
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

def vectorize_colors(colors):
	"""
	Convert color data into a vector format
	:param colors: A list containing only the characters {W, U, B, R, G}
	:return: A one-hot vector [W U B R G]
	"""
	vector = [1 if 'W' in colors else 0,
	          1 if 'U' in colors else 0,
	          1 if 'B' in colors else 0,
	          1 if 'R' in colors else 0,
	          1 if 'G' in colors else 0]
	return np.array(vector, dtype=np.float64)

def pre_process_data(dataset, unique_subtypes=None, keyword_categories=None, min_subtypes=30, keyword_N=15):
	"""
	Convert the X data into feature vectors
	:param dataset: X values
	:param unique_subtypes: If previously called for training, pass in the output to maintain the same feature vector
	:param keyword_categories: If previously called for training, pass in the output to maintain the same feature vector
	:param min_subtypes: The minimum number of subtypes to be included in the one-hot types vector
	:param keyword_N: The number of each type of keyword to include in the feature vector (total of 2 * keyword_N)
	:return card_vectors: The converted feature vectors for training, an nd-array
	:return unique_subtypes: For maintaining the same feature vector
	:return keyword_categories: For maintaining the same feature vector
	"""
	all_text = dataset.loc[:, 'text']

	if unique_subtypes == None:
		unique_subtypes = Counter()
		for l in dataset.loc[:, 'subtypes'].values:
			unique_subtypes.update(l)
		unique_subtypes = Counter(el for el in unique_subtypes.elements() if unique_subtypes[el] >= min_subtypes) # hyperparam
		unique_subtypes = list(set(unique_subtypes.elements()))

	keyword_vectors, keyword_categories = tag_keywords(dataset, keyword_categories, top_N=keyword_N) # The keyword_categories are messing state up
	card_vectors = []

	for name, text in all_text.items():
		if text is None: text = ''
		vector = []
		non_text_features = dataset.loc[name]

		for subtype in unique_subtypes:
			vector.append(1 if subtype in non_text_features.loc['subtypes'] else 0)

		vector.append(1 if 'Artifact' in non_text_features.loc['types'] else 0)
		vector.append(1 if 'Enchantment' in non_text_features.loc['types'] else 0)

		non_text_features = non_text_features.drop(['text', 'types', 'subtypes'])

		for feature in non_text_features.values:
			vector.append(int(feature) if feature != '*' else 0)
		card_vectors.append(vector)

	card_vectors = np.array([np.array(xi, dtype=np.float64) for xi in card_vectors], dtype=np.float64)
	card_vectors = np.concatenate((card_vectors, keyword_vectors), axis=1)

	return card_vectors, unique_subtypes, keyword_categories

def tag_keywords(dataset, common_tokens=None, top_N=15):
	"""
	Create a feature vector of all relevant abilities and add the feature to our dataset
	:param dataset: a DataFrame containing the rows 'abilityWords', 'keywordAbilities' and 'keywordActions'
	:return token_vectors: a M x (top_N * 2) list of arrays
	:return keyword_categories: the top_N * 2 length list of the keywords represented in the token_vectors
	"""
	all_text = dataset.loc[:, 'text']

	if common_tokens is None:
		with open('data/Keywords.json') as f:
			keywords = json.load(f)
		keywordAbilities = pd.Series(keywords['keywordAbilities'])
		keywordActions = pd.Series(keywords['keywordActions'])
		keywordAbilitiesFreq = Counter()
		keywordActionsFreq = Counter()

		for name, text in all_text.items():
			if text is None:
				continue
			text = re.sub("[\(\[].*?[\)\]]", "", text).replace('\n', ' ') # strip reminder text

			for token in keywordAbilities:
				if token in text.split(' '): keywordAbilitiesFreq.update([token])
			for token in keywordActions:
				if token in text.split(' '): keywordActionsFreq.update([token])

		common_tokens = keywordAbilitiesFreq.most_common(top_N) + keywordActionsFreq.most_common(top_N)
		common_tokens = [x[0] for x in common_tokens]

	token_vectors = []
	for name, text in all_text.items():
		if text is None:
			token_vectors.append(np.zeros(len(common_tokens)))
		else:
			vector = []
			re_text = re.split(r'\W', text)
			for common_token in common_tokens:
				vector.append(re_text.count(common_token)) if re_text is not None else 0
			token_vectors.append(np.array(vector, dtype=np.float64))
	token_vectors = np.array(token_vectors, dtype=np.float64)
	return token_vectors, common_tokens

def load_data():
	dataset = pd.read_json('data/CleanedDataset.json')
	dataset.append(pd.Series(name='TextVector'))
	Y = dataset.loc['colors'].apply(vectorize_colors)
	Y = np.array([np.array(yi, dtype=np.float64) for yi in Y], dtype=np.float64)
	X = dataset.drop('colors').transpose()
	return X, Y

def time_classifier(X, Y, clf, min_subtypes=25, keyword_N=50, output_file=None, output_console=False):
	card_vectors_train, unique_subtypes, keyword_vector = pre_process_data(X, min_subtypes=min_subtypes,
	                                                                       keyword_N=keyword_N)
	scores = pd.DataFrame([], columns=['train_time', 'test_time'])

	i = 0
	for train_index, test_index in KFold(n_splits=5, shuffle=True).split(X):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		card_vectors_train, unique_subtypes, keyword_vector = pre_process_data(X_train, min_subtypes=min_subtypes,
		                                                                       keyword_N=keyword_N)
		train_start = time.time()
		clf.fit(card_vectors_train, Y_train)
		train_end = time.time()
		Y_pred_train = clf.predict(card_vectors_train)

		card_vectors_test, _, _ = pre_process_data(X_test, unique_subtypes, keyword_vector)
		test_start = time.time()
		Y_pred_test = clf.predict(card_vectors_test)
		test_end = time.time()

		scores.loc[i] = [train_end - train_start, test_end - test_start]
		i += 1
	if output_console: print(scores)
	if output_file is not None:
		scores.to_json(output_file)

def test_classifier(X, Y, clf, min_subtypes=25, keyword_N=50, output_file=None, output_console=False):
	print(f'Beginning test of classifier with output_file={output_file}')
	card_vectors_train, unique_subtypes, keyword_vector = pre_process_data(X, min_subtypes=min_subtypes,
	                                                                       keyword_N=keyword_N)

	scores = pd.DataFrame([], columns=['accuracy_score', 'hamming_loss', 'f1_score', 'precision_score', 'recall_score'])

	i = 0
	for train_index, test_index in KFold(n_splits=5, shuffle=True).split(X):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		card_vectors_train, unique_subtypes, keyword_vector = pre_process_data(X_train, min_subtypes=min_subtypes, keyword_N=keyword_N)

		clf.fit(card_vectors_train, Y_train)
		Y_pred_train = clf.predict(card_vectors_train)

		card_vectors_test, _, _ = pre_process_data(X_test, unique_subtypes, keyword_vector)
		Y_pred_test = clf.predict(card_vectors_test)

		test_accuracy_score = accuracy_score(Y_test, Y_pred_test)
		test_hamming_loss = hamming_loss(Y_test, Y_pred_test)
		test_f1_score = f1_score(Y_test, Y_pred_test, average='samples')
		test_precision_score = precision_score(Y_test, Y_pred_test, average='samples')
		test_recall_score = recall_score(Y_test, Y_pred_test, average='samples')

		scores.loc[i] = [test_accuracy_score, test_hamming_loss, test_f1_score, test_precision_score, test_recall_score]
		i+=1
	if output_console: print(scores)
	scores.to_json(output_file)

def hyperparameter_tuning():
	card_vectors_train, unique_subtypes, keyword_vector = pre_process_data(X, min_subtypes=min_subtypes,
	                                                                       keyword_N=keyword_N)
	parameters = {'hidden_layer_sizes': [75, 100, 125, 150], 'alpha': [0.00005, 0.0001, 0.0002, 0.0005], 'max_iter': [2000]}
	c = MLPClassifier()
	clf = GridSearchCV(c, parameters, cv=5)
	print("Calculating best parameters...")
	clf.fit(card_vectors_train, Y)
	print(f"Best params= {clf.best_params_}")
	print(f"Best score= {clf.best_score_}")

def time_classifier_multi(X, Y, clf, output_file_base, min_subtypes=25, keyword_N=50):
	print(f"Beginning time test of clf output to file_base={output_file_base}")
	for split in [25, 50, 75, 100]: # 4 segments
		time_classifier(X.iloc[:int(X.shape[0] * (split / 100))], Y, clf, output_file=f'{output_file_base}_{split}.json')

def mtg_color(min_subtypes=25, keyword_N=50):
	X, Y = load_data()

	test_classifier(X, Y, RandomForestClassifier(n_estimators=100, max_depth=None), output_file='Results/random_forest_results.json')
	test_classifier(X, Y, MLPClassifier(alpha=0.0001, hidden_layer_sizes=150, max_iter= 2000), output_file='Results/mlp_classifier_results.json')
	test_classifier(X, Y, OneVsRestClassifier(LinearSVC(C=4, max_iter=200000)), output_file='Results/linear_svc_results.json')

	time_classifier_multi(X, Y, RandomForestClassifier(n_estimators=100, max_depth=None), output_file_base='Results/random_forest_times')
	time_classifier_multi(X, Y, MLPClassifier(alpha=0.0001, hidden_layer_sizes=150, max_iter= 2000), output_file_base='Results/mlp_classifier_times')
	time_classifier_multi(X, Y, OneVsRestClassifier(LinearSVC(C=4, max_iter=200000)), output_file_base='Results/linear_svc_times')

if __name__ == "__main__":
	mtg_color()