import random

proportion = 0.1
with open('training.csv', 'r') as fin:
    lines = fin.readlines()
    train = open('train.csv', 'w')
    valid = open('valid.csv', 'w')
    random.shuffle(lines)
    train.writelines(lines[:int((1 - proportion) * len(lines))])
    valid.writelines(lines[int((1 - proportion) * len(lines)):])

    train.close()
    valid.close()
