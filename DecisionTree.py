import csv
import math
import random


class DecisionTree():
    tree = {}

    def learn(self, training_set, attributes, target):
        self.tree = build_tree(training_set, attributes, target)


# Class Node which will be used while classify a test-instance using the tree which was built earlier
class Node():
    value = ""
    children = []

    def __init__(self, val, dictionary):
        self.value = val
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()


# Majority Function which tells which class has more entries in given data-set
def majorClass(attributes, data, target):
    freq = {}
    index = attributes.index(target)
    for tuple in data:
        k = tuple[index]
        if k in freq.keys():
            freq[tuple[index]] += 1
        else:
            freq[tuple[index]] = 1

    max = 0
    major = ""

    for key in freq.keys():
        if freq[key] > max:
            max = freq[key]
            major = key

    return major


# Calculates the entropy of the data given the target attribute
def entropy(attributes, data, targetAttr):
    freq = {}
    dataEntropy = 0.0

    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        i = i + 1

    # i = i - 1
    for entry in data:
        if entry[i] in freq:
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]] = 1.0

    for freq in freq.values():
        dataEntropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return dataEntropy


# Calculates the information gain (reduction in entropy) in the data when a particular attribute is chosen for splitting the data.
def info_gain(attributes, data, attr, targetAttr):
    freq = {}
    subsetEntropy = 0.0
    childEntropy = 0.0
    i = attributes.index(attr)

    for entry in data:
        if entry[i] in freq:
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]] = 1.0

    for val in freq.keys():
        valProb = freq[val] / sum(freq.values())
        dataSubset = [entry for entry in data if entry[i] == val]
        subsetEntropy = valProb * entropy(attributes, dataSubset, targetAttr)
        childEntropy += subsetEntropy

    parentEntropy = entropy(attributes, data, targetAttr)
    return (parentEntropy - childEntropy)


# This function chooses the attribute among the remaining attributes which has the maximum information gain.
def attr_choose(data, attributes, target):
    best = attributes[1]
    maxGain = 0;

    for attr in attributes:
        if attr != target:
            newGain = info_gain(attributes, data, attr, target)
            if newGain > maxGain:
                maxGain = newGain
                best = attr

    return best


def get_unique_values(data, attributes, attr):
    ind = attributes.index(attr)
    values = []

    for entry in data:
        if entry[ind] not in values:
            values.append(entry[ind])

    return values


def get_data_foruniquevalues(data, attributes, best, value):
    new_data = [[]]
    index = attributes.index(best)

    for entry in data:
        if entry[index] == value:
            newentry = []
            for i in range(0, len(entry)):
                if (i != index):
                    newentry.append(entry[i])
            new_data.append(newentry)
    new_data.remove([])

    return new_data


def build_tree(data, attributes, target):
    data = data[:]
    vals = [record[attributes.index(target)] for record in data]
    default = majorClass(attributes, data, target)

    if not data or (
            len(attributes) - 1) <= 0:  # If there is no data or if it has only two attributes (class and parameter)
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]  # if all belong to one classvals
    else:
        best = attr_choose(data, attributes, target)
        tree = {best: {}}
        # print('Best Attribute for', best)

        for val in get_unique_values(data, attributes, best):
            new_data = get_data_foruniquevalues(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            # print('parent',best)
            # print('child', val)
            # print('\n')
            subtree = build_tree(new_data, newAttr, target)
            tree[best][val] = subtree

        return tree


def run_decision_tree():
    data = []
    test = []

    with open("MushroomTrain.csv") as tsv:
        for line in csv.reader(tsv, delimiter=","):
            data.append(tuple(line))

    with open("MushroomTest.csv") as tsv:
        for line in csv.reader(tsv, delimiter=","):
            test.append(tuple(line))

    attributes = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises']
    target = attributes[0]

    tree = DecisionTree()
    tree.learn(data, attributes, target)
    results = []
    accuracy = 0.0

    # test data set and compare the classification accuracy on this test set with the one on the training set.
    for entry in test:
        tempDict = tree.tree.copy()
        result = ""
        while (isinstance(tempDict, dict)):
            rootkey = ""
            rootvalue = {}

            for key, val in tempDict.items():
                rootkey = key
                rootvalue = val

            root = Node(rootkey, rootvalue)
            index = attributes.index(root.value)
            value = entry[index]

            tempDict = rootvalue
            if (value in tempDict.keys()):
                child = Node(value, tempDict[value])
                result = tempDict[value]
                tempDict = tempDict[value]
            else:
                result = "Null"
                break

        if (result != "Null"):
            results.append(result == entry[0])

    accuracy = float(results.count(True)) / float(len(test))

    print('Accuracy for MushroomTest Dataset = ', accuracy)

    K = 4
    acc = []
    for k in range(K):
        random.shuffle(data)
        result_fold = []
        training_set = [x for i, x in enumerate(data) if i % K != k]
        test_set = [x for i, x in enumerate(data) if i % K == k]
        accuracy = 0.0
        tree = DecisionTree()
        tree.learn(training_set, attributes, target)

        # test data set and compare the classification accuracy on this test set with the one on the training set.
        for entry in test_set:
            tempDict = tree.tree.copy()
            result = ""
            while (isinstance(tempDict, dict)):
                rootkey = ""
                rootvalue = {}

                for key, val in tempDict.items():
                    rootkey = key
                    rootvalue = val

                root = Node(rootkey, rootvalue)
                index = attributes.index(root.value)
                value = entry[index]

                tempDict = rootvalue
                if (value in tempDict.keys()):
                    child = Node(value, tempDict[value])
                    result = tempDict[value]
                    tempDict = tempDict[value]
                else:
                    result = "Null"
                    break

            if (result != "Null"):
                result_fold.append(result == entry[0])

        accuracy = float(result_fold.count(True)) / float(len(test_set))
        acc.append(accuracy)
        print('Accuracy = {0} for fold {1}'.format(accuracy, k))

    avg_acc = sum(acc) / len(acc)
    print('Average accuracy: {0}'.format(avg_acc))


run_decision_tree()