from __future__ import division
from collections import Counter
import math
from node import Node
from scipy import stats
from scipy.stats import chi2

label_index = 16
num_classifications = 26

##### Main functions for tree construction

# Returns the entropy of a set of examples
def entropy(examples):
    if len(examples) == 0:
        return 0
    counter = Counter(example[label_index] for example in examples)
    if True in (counter[i] == len(examples) for i in range(0, num_classifications)):
        return 0 # if all examples fall into same catagory
    else:
        return sum(-counter[i] / len(examples) * log_base_2(counter[i] / len(examples)) for i in range(0, num_classifications))

# Returns the information gain by splitting on the examples on the given attribute
def information_gain(examples, attribute):
    partition_map = partition_on_attribute(examples, attribute)
    expected_entropy = 0
    for key in partition_map:
        expected_entropy += (len(partition_map[key]) / len(examples)) * entropy(partition_map[key])
    return entropy(examples) - expected_entropy


# Returns the root node of a decision tree constructed using the given set of examples and attributes
def construct_tree(examples, attributes, threshold):
    if entropy(examples) == 0:
        return Node(classification=examples[0][label_index], examples=examples)
    elif len(attributes) == 0:
        counter = Counter(example[label_index] for example in examples)
        return Node(classification=max((counter[key], key) for key in counter)[1], examples=examples)
    else:
        best_attribute = get_best_attribute(examples, attributes)
        m, k = degrees_of_freedom(examples, best_attribute)
        s = s_value(m, k, examples, best_attribute)
        df = (m - 1) * (k - 1)
        p_value = 1 - chi2.cdf(s, df)
        if p_value > threshold:
            counter = Counter(example[label_index] for example in examples)
            return Node(classification=max((counter[key], key) for key in counter)[1], examples=examples)
        attributes.remove(best_attribute)
        root = Node(attribute=best_attribute, examples=examples)
        partition_map = partition_on_attribute(examples, best_attribute)
        for attribute_value in partition_map:
            partition = partition_map[attribute_value]
            if len(partition) == 0:
                counter = Counter(example[label_index] for example in examples)
                root.children[attribute_value] = Node(classification=max((counter[key], key) for key in counter)[1], examples=examples)
            else:
                root.children[attribute_value] = construct_tree(partition, attributes, threshold)
        attributes.append(best_attribute)
        return root

##### Helper functions and functions for classifying new data

# Returns a map from each possible value of attribute to the subset of examples that have that particular value
def partition_on_attribute(examples, attribute):
    return {attribute_value: [example for example in examples if example[attribute] == attribute_value] for attribute_value in range(0, 16)}

# Returns the attribute with the highest information gain
def get_best_attribute(examples, attributes):
    return max((information_gain(examples, attribute), attribute) for attribute in attributes)[1]

# Returns the log base two of a value with special case log(0) = 0
def log_base_2(val):
    if (val == 0):
        return 0
    else:
        return math.log(val, 2)

# Reads from a feature file and label file and outputs a list of tuples representing training examples
def get_training_examples(feature_file, label_file):
    examples = []
    features = open(feature_file, 'r')
    labels = open(label_file, 'r')
    feature_line = features.readline()
    label_line = labels.readline()
    while feature_line and label_line:
        examples.append(tuple(map(int, feature_line.split() + [label_line])))
        feature_line = features.readline()
        label_line = labels.readline()
    features.close()
    labels.close()
    return examples

# Attempts to classify a set of test data and outputs the results to a file
def classify_file(feature_file, predicted_label_file, correct_label_file, root):
    diff_count = 0
    correct_count = 0
    predicted_labels = open(predicted_label_file, 'w')
    test_features = open(feature_file, 'r')
    test_labels = open(correct_label_file, 'r')
    feature_line = test_features.readline()
    label_line = test_labels.readline()
    while feature_line and label_line:
        example = map(int, feature_line.split())
        correct_label = int(label_line)
        predicted_label = classify_example(root, example, correct_label)
        if (predicted_label != correct_label):
            diff_count += 1
        else:
            correct_count += 1
        predicted_labels.write(str(predicted_label) + '\n')
        feature_line = test_features.readline()
        label_line = test_labels.readline()
    test_features.close()
    test_labels.close()
    print 'Pecent correctly classified: ' + str((correct_count / (diff_count + correct_count)))
    return (diff_count, correct_count)

# Returns the leaf nodes of a tree
def get_leaves(root, leaves):
    if root.children == {}:
        leaves.append(root)
    else:
        for key in root.children:
            get_leaves(root.children[key], leaves)

# Counts the nodes in a tree
def count_nodes(root):
    if root.children == {}:
        return 1
    else:
        result = 0
        for key in root.children:
            result += count_nodes(root.children[key])
        return result + 1

##### Functions related to chi-squared checks

# Classifies a novel example using the provided decision tree
def classify_example(root, example, correct_label):
    if root.classification is not None:
        root.hit_count += 1
        if root.classification == correct_label:
            root.num_correct += 1
        return root.classification
    else:
        split_attribute = root.attribute
        attribute_value = example[split_attribute]
        return classify_example(root.children[attribute_value], example, correct_label)

# Returns the number of examples with the given attribute value and classification
def calculate_observed(examples, attribute, attribute_value, classification):
    count = 0
    for example in examples:
        if example[attribute] == attribute_value and example[16] == classification:
            count += 1
    return count

# Returns the expected count for the given class when A = attribute_value
def calculate_expected(examples, attribute, attribute_value, classification):
    count_with_attribute_value = 0
    count_with_classification = 0
    for example in examples:
        if example[attribute] == attribute_value:
            count_with_attribute_value += 1
        if example[16] == classification:
            count_with_classification += 1
    return count_with_attribute_value * (count_with_classification / len(examples))

# Returns the value s in the specification equation
def s_value(m, k, examples, attribute):
    result = 0
    for j in range(1, m):
        for i in range(1, k):
            observed = calculate_observed(examples, attribute, j, i)
            expected = calculate_expected(examples, attribute, j, i)
            if (expected != 0):
                result += math.pow(observed - expected, 2) / expected
    return result

# Returns the m and k values for degress of freedom
def degrees_of_freedom(examples, attribute):
    # partitions = partition_on_attribute(examples, attribute)
    # distinct_attribute_values = 0
    # for key in partitions:
    #     if (len(partitions[key]) > 0):
    #         distinct_attribute_values += 1
    counter = Counter([example[16] for example in examples])
    distinct_classifications = len(counter)
    result = (16, distinct_classifications)
    return result


# examples = get_training_examples('training-features.txt', 'training-labels.txt')
# root = construct_tree(examples, range(0, 16), 1)
# print 'num nodes: ' + str(count_nodes(root))
# classify_file('test-features.txt', 'predicted-labels.txt', 'test-labels.txt', root)
#classify_file('training-features.txt', 'predicted-labels.txt', 'training-labels.txt', root)
# leaves = []
# get_leaves(root, leaves)
# top_leaves = sorted(leaves, key=lambda leaf: leaf.hit_count, reverse=True)[:5]
# for leaf in top_leaves:
#     print 'hit count: ' + str(leaf.hit_count) + ', num correct: ' + str(leaf.num_correct) + ', classification: ' + str(leaf.classification)
