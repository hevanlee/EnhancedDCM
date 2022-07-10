from keras import backend as K
from five_parallel_synth import data_manager as fiveDM
if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.append(dir(path[0]))
    splits = path[0].split('/')

    parent = '/'.join(splits[:-1])
    path.append(dir(parent))

import utilities.run_utils as ru
import argparse
"""
	Run script for fiveParallel Real Datset Experiments
	For each model: 
		- Define architecture (# X inputs, # Q inputs, model architecture)
		- Create input with keras_input()
		- Run the appropriate function below
	
	Main() flags:
	------------
	models:		Estimates many models on fiveParallel dataset
	scan: 		Perform a architectural scan of neurons on L-MNL
"""


parser = argparse.ArgumentParser(description='Choose Flags for training on experiments')
parser.add_argument('--scan', action='store_true', help='Trains multiple L-MNL models of increasing size on fiveParallel')
parser.add_argument('--models', action='store_true', help='Trains a full set of models on fiveParallel')
args = parser.parse_args()

models = args.models
scan = args.scan

choices_num = 4
batchSize = 50


def fiveParallelMNL(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart='', saveName=''):
	nEpoch = 120

	betas, saveExtension = ru.runMNL(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
									 filePart, saveName=saveName)
	K.clear_session()
	return betas, saveExtension


def fiveParallelNN(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=False,
				 filePart='', saveName='',
				 networkSize=100):
	nEpoch = 200

	saveExtension = ru.runNN(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
							 extraInput, nExtraFeatures, filePart, saveName=saveName, networkSize=networkSize)
	K.clear_session()
	return saveExtension


def fiveParallelMixed(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=False,
					minima=None, train_betas=True, filePart='', saveName='',
					networkSize=100, hidden_layers=1, verbose=0):
	nEpoch = 200

	betas, saveExtension = ru.runMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name,
									   batchSize, extraInput, nExtraFeatures, minima, train_betas, filePart,
									   saveName=saveName, networkSize=networkSize, hidden_layers=hidden_layers,
									   verbose=verbose)
	K.clear_session()
	return betas, saveExtension

def fiveParallelHrusch(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart='', saveName=''):
	nEpoch = 200

	saveExtension = ru.runHrusch(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
									 filePart, saveName=saveName)
	K.clear_session()
	return saveExtension

def fiveParallelHrusch07(filePath, fileInputName, beta_num, choices_num, train_data_name, filePart='', saveName=''):
	nEpoch = 200

	saveExtension = ru.runHrusch07(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name, batchSize,
									 filePart, saveName=saveName)
	K.clear_session()
	return saveExtension


if __name__ == '__main__':

	# splits data into train and test set
	# extensions = swissDM.train_test_split(filePath, seed = 32)

	filePath = 'five_parallel_synth/'
	extensions = ['_train', '_test']

	if models:
		folderName = 'models/'
		fileInputName = 'five_parallel_synth'

		'''
		print("Full MNL")
		simpleArchitecture = False
		beta_num = 9
		nExtraFeatures = 8
		_, _, train_data_name = swissDM.keras_input(filePath+folderName, fileInputName, filePart=extensions[0],
													simpleArchitecture=simpleArchitecture)
		fiveParallelMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name,
					  filePart=extensions[0], saveName='_Full')
		'''
		print("L-MNL Naive")
		fiveParallelMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name,
						extraInput=True, saveName="_Naive")

		print("L-MNL")
		lmnlArchitecture = True
		beta_num = 3
		nExtraFeatures = 12
		_, _, train_data_name = swissDM.keras_input(filePath+folderName, fileInputName, filePart=extensions[0],
													lmnlArchitecture=lmnlArchitecture)
		fiveParallelMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True)
		'''
		print("MNL")
		beta_num = 5
		simpleArchitecture = True
		_, _, train_data_name = swissDM.keras_input(filePath+folderName, fileInputName, filePart=extensions[0],
													simpleArchitecture=simpleArchitecture)
		fiveParallelMNL(filePath+folderName, fileInputName,  beta_num, choices_num, train_data_name, filePart=extensions[0])
		'''
	if scan:
		folderName = 'scan/'
		fileInputName = 'five_parallel_synth'
		list = [1, 5, 10, 15, 25, 50, 100, 200, 500, 1001, 2000, 5000]
		for i in list:
			print("L-MNL with {} neurons".format(i))
			lmnlArchitecture = True
			beta_num = 5
			nExtraFeatures = 12
			_, _, train_data_name = fiveDM.keras_input(filePath+folderName, fileInputName, filePart=extensions[0],
														simpleArchitecture=lmnlArchitecture)
			_, saveExtension = FiveParallelMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures,
											   train_data_name, extraInput=True, saveName='{}'.format(i),
											   filePart=extensions[0], networkSize=i)
			