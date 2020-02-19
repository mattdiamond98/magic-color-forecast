import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_confidence_intervals():
	rf  = pd.read_json('Results/random_forest_results.json')
	svc = pd.read_json('Results/linear_svc_results.json')
	mlp = pd.read_json('Results/mlp_classifier_results.json')

	rf_mean  = rf ['accuracy_score'].mean()
	svc_mean = svc['accuracy_score'].mean()
	mlp_mean = mlp['accuracy_score'].mean()
	rf_std   = rf ['accuracy_score'].std()
	svc_std  = svc['accuracy_score'].std()
	mlp_std  = mlp['accuracy_score'].std()

	x1 = min(rf_mean, svc_mean, mlp_mean) - 2.5 * (max(rf_std, svc_std, mlp_std))
	x2 = max(rf_mean, svc_mean, mlp_mean) + 2.5 * (max(rf_std, svc_std, mlp_std))
	x = np.arange(x1, x2, 0.0001)

	rf_y  = norm.pdf(x, rf_mean, rf_std)
	svc_y = norm.pdf(x, svc_mean, svc_std)
	mlp_y = norm.pdf(x, mlp_mean, mlp_std)

	fig, ax = plt.subplots(figsize=(9, 6))
	ax.plot(x, rf_y, label='random forests classifier')
	ax.plot(x, svc_y, label='support vector classifer')
	ax.plot(x, mlp_y, label='multilayer perceptron classifier')
	ax.set_xlabel('accuracy_score')
	ax.set_yticklabels([])
	ax.set_title('Performance Comparison')
	plt.legend()
	plt.grid(True)
	plt.savefig('Results/algorithm_comparison.png', dpi=72, bbox_inches='tight')
	plt.show()

def concatenate_time_data(file_base):
	train = {}
	test = {}
	for time in [25, 50, 75, 100]:
		file_name = f'{file_base}_{time}.json'
		avg_times = pd.read_json(file_name).mean()
		train[time] = avg_times['train_time']
		test[time]  = avg_times['test_time']

	return pd.Series(train), pd.Series(test)

def get_mean_results(file):
	results = pd.read_json(file)
	print(results.mean(axis=0))

def plot_time_comparisons():
	rf_train_times, rf_test_times = concatenate_time_data('Results/random_forest_times')
	svc_train_times, svc_test_times = concatenate_time_data('Results/linear_svc_times')
	mlp_train_times, mlp_test_times = concatenate_time_data('Results/mlp_classifier_times')

	x = [25, 50, 75, 100]

	fig, ax = plt.subplots(figsize=(9, 6))
	ax.plot(x, rf_train_times, label='random forests classifier')
	ax.plot(x, svc_train_times, label='support vector classifer')
	ax.plot(x, mlp_train_times, label='multilayer perceptron classifier')
	ax.set_xlabel('percent of training data')
	ax.set_ylabel('average train time')

	plt.legend()
	plt.grid(True)
	plt.xticks(x)
	plt.yticks([0, 20, 40, 60])
	plt.title('Training Time Comparison')
	plt.savefig('Results/time_comparison.png', dpi=72, bbox_inches='tight')
	plt.show()

if __name__ == "__main__":
	print('Random forests results:')
	get_mean_results('Results/random_forest_results.json')
	print('Linear SVC results:')
	get_mean_results('Results/linear_svc_results.json')
	print('MLP classifier results:')
	get_mean_results('Results/mlp_classifier_results.json')
	plot_confidence_intervals()
	plot_time_comparisons()