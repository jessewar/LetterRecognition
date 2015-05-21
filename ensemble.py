from collections import Counter
import random
import decision_tree

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

# Samples n items, with replacement, from examples
def sample(examples, n):
    result = []
    for i in range(n):
        random_index = random.randint(0, len(examples) - 1)
        result.append(examples[random_index])
    return result

# Attempts to classify a set of test data and outputs the results to a file
def ensemble_classify_file(test_feature_file, test_label_file, trees):
    diff_count = 0.0
    correct_count = 0.0
    test_features = open(test_feature_file, 'r')
    test_labels = open(test_label_file, 'r')
    feature_line = test_features.readline()
    label_line = test_labels.readline()
    while feature_line and label_line:
        example = map(int, feature_line.split())
        correct_label = int(label_line)
        counter = Counter()
        for i in range(len(trees)):
            predicted_label = ensemble_classify_example(trees[i], example, correct_label)
            counter[predicted_label] += 1
        majority_label = max(counter, key=counter.get)
        if (majority_label != correct_label):
            diff_count += 1
        else:
            correct_count += 1
        feature_line = test_features.readline()
        label_line = test_labels.readline()
    test_features.close()
    test_labels.close()
    print 'Pecent correctly classified: ' + str((correct_count / (diff_count + correct_count)))
    return (diff_count, correct_count)

# Classifies a novel example using the provided decision tree
def ensemble_classify_example(root, example, correct_label):
    if root.classification is not None:
        root.hit_count += 1
        if root.classification == correct_label:
            root.num_correct += 1
        return root.classification
    else:
        split_attribute = root.attribute
        attribute_value = example[split_attribute]
        return ensemble_classify_example(root.children[attribute_value], example, correct_label)

def get_bagging_classifier(num_samples, examples):
    trees = []
    for i in range(num_samples):
        training_set = sample(examples, len(examples))
        root = decision_tree.construct_tree(training_set, range(0, 16), 1)
        trees.append(root)
    return trees


examples = get_training_examples('training-features.txt', 'training-labels.txt')
classifier = get_bagging_classifier(10, examples)
ensemble_classify_file('test-features.txt', 'test-labels.txt', classifier)


