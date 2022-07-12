import numpy as np
import random
import shelve
import _pickle as pickle
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, Add, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.losses import mean_squared_error
import os


"""
    This script adapts the five_parallel_synth dataset to train on the keras models
"""



fileName = 'five_parallel_synth'


def train_test_split(filePath, seed = None):
    """ Shuffles the dataset and splits in test/train set """

    if seed is not None:
        random.seed(seed)

    lines = open(filePath + fileName + '.dat', 'r').readlines()

    data = np.loadtxt(filePath + fileName + '.dat',skiprows = 1)
    TAG = data[:,-1]

    LF = [p for p,i in enumerate(TAG) if i == 1]
    CA = [p for p,i in enumerate(TAG) if i == 2]
    GR = [p for p,i in enumerate(TAG) if i == 3]
    Other = [p for p,i in enumerate(TAG) if i == 4]


    np.random.shuffle(LF)
    np.random.shuffle(CA)
    np.random.shuffle(GR)
    np.random.shuffle(Other)


    test_ratio = 0.8
    step_LF = int(len(LF)*test_ratio)
    step_CA = int(len(CA)*test_ratio)
    step_GR = int(len(CAR)*test_ratio)
    step_Other = int(len(Other)*test_ratio)

    file_part = open(filePath + fileName + '_train' + '.dat', 'w')
    file_part.writelines(lines[0])
    part = [lines[i+1] for i in LF[:step_LF]]
    file_part.writelines(part)
    part = [lines[i+1] for i in CA[:step_CA]]
    file_part.writelines(part)
    part = [lines[i+1] for i in GR[:step_GR]]
    file_part.writelines(part)
    part = [lines[i+1] for i in Other[:step_Other]]
    file_part.writelines(part)
    file_part.close()

    file_part = open(filePath + fileName + '_test' + '.dat', 'w')
    file_part.writelines(lines[0])
    part = [lines[i+1] for i in LF[step_LF:]]
    file_part.writelines(part)
    part = [lines[i+1] for i in CA[step_CA:]]
    file_part.writelines(part)
    part = [lines[i+1] for i in GR[step_GR:]]
    file_part.writelines(part)
    part = [lines[i+1] for i in Other[step_Other:]]
    file_part.writelines(part)
    file_part.close()

    return ['_train', '_test']



def keras_input(filePath, fileInputName, filePart = '', simpleArchitecture = False, lmnlArchitecture = False, write = True):
    """
    Prepares Input for Models. Based on Dataset, utility functions and number of alternatives
    The first input is the X feature set, it ressembles the utility functions.
        - The shape is (n x betas+1 x alternatives), where the added +1 is the label.
    The second input is the Q feature set.
        - The shape is (n x Q_features x 1)
    :param filePath:        path to dataset
    :param fileInputName:   name of dataset
    :param filePart:        dataset extension (e.g. _train, _test)
    :param simpleArchitecture:  Smaller Utility Function
    :param lmnlArchitecture:    L-MNL Utility Function (Small and no ASC)
    :param write:           Save X and Q inputs in a .npy
    :return:    train_data: X inputs Table with Choice label,
                extra_data: Q inputs vector
                train_data_name: saved name to X's .npy file
    """

    extend = ''
    if simpleArchitecture:
        extend = '_simple'
    if lmnlArchitecture:
        extend = '_noASC'

    train_data_name = filePath+ 'keras_input_' + fileInputName + extend + filePart + '.npy'

    #filePath = 'swissmetro_paper/'
    filePath = os.path.dirname(os.path.realpath(__file__))+'/'
    data = np.loadtxt(filePath + fileName + filePart + '.dat',skiprows = 1)
    beta_num = 5
    choices_num = 4

    #exclusions:
    ID = data[:, 0]

    exclude = (ID == 0) > 0
    exclude_list = [i for i, k in enumerate(exclude) if k > 0]

    data = np.delete(data,exclude_list, axis = 0)

    #Define:
    TAG = data[:,-1]
    p1 = data[:, 0:42]
    p2 = data[:, 42:84]
    p3 = data[:, 84:126]
    p4 = data[:, 126:168]
    p5 = data[:, 168:210]

    scale = 100.0
    #scale = 1.0

    p1_scaled = p1/scale
    p2_scaled = p2/scale
    

    ASCs = np.ones(TAG.size)
    ZEROs = np.zeros(TAG.size)

    TAG_LF = (TAG == 1)
    TAG_CA = (TAG == 2)
    TAG_GR = (TAG == 3)
    TAG_Other = (TAG == 4)

    #  lmnl only
    train_data = np.array(
            [[p1, p2, p3, p4, TAG_LF],
            [p1, p2, p3, p4, TAG_CA],
            [p1, p2, p3, p4, TAG_GR]
            [p1, p2, p3, p4, TAG_Other]])
    '''
    if simpleArchitecture:    
        train_data = np.array(
            [[ZEROs, ZEROs, TRAIN_TT_SCALED, TRAIN_COST_SCALED, TRAIN_HE_SCALED, CHOICE_TRAIN],
            [ZEROs,  ASCs,  SM_TT_SCALED,    SM_COST_SCALED,    SM_HE_SCALED,    CHOICE_SM],
            [ASCs,   ZEROs, CAR_TT_SCALED,   CAR_CO_SCALED,     ZEROs,    CHOICE_CAR]] )
    if lmnlArchitecture:
        train_data = np.array(
            [[p1, p2, p3, p4, TAG_LF],
            [p1, p2, p3, p4, TAG_CA],
            [p1, p2, p3, p4, TAG_GR]
            [p1, p2, p3, p4, TAG_Other]])
    '''
    train_data = np.swapaxes(train_data,0,2)

    delete_list = range(len(data))
    delete_list = np.delete(delete_list, [2, 3, 4])

    if simpleArchitecture or lmnlArchitecture:
        # Hybrid Simple
        extra_data = np.delete(data, delete_list, axis = 1)
    else:
        # Hybrid MNL
        extra_data = np.delete(data, range(len(data)), axis = 1)


    if write:
        np.save(train_data_name, np.array(train_data, dtype=np.float32))
        np.save(train_data_name[:-4] + '_extra.npy', extra_data)

    return train_data, extra_data, train_data_name