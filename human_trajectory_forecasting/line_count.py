import numpy as np

filePath = 'five_parallel_synth/'
fileName = 'five_parallel_synth'
data = open(filePath + fileName + '.dat', 'r').readlines()

expected_len = 212
lengths = []

for line in data:
    if len(line.split('\t')) != expected_len:
        split = line.split('\t')
        lengths.append((split[0], len(split)))
print("line lengths: ", lengths)