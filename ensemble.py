from bagging_classifier import BaggingClassifier

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

# Uses a BaggingClassifier instance to classify novel examples and outputs a success rate
def ensemble_classify_file(test_feature_file, test_label_file, classifier):
    diff_count = 0.0
    correct_count = 0.0
    test_features = open(test_feature_file, 'r')
    test_labels = open(test_label_file, 'r')
    feature_line = test_features.readline()
    label_line = test_labels.readline()
    while feature_line and label_line:
        example = map(int, feature_line.split())
        correct_label = int(label_line)
        majority_label = classifier.classify_example(example, correct_label)
        if (majority_label != correct_label):
            diff_count += 1
        else:
            correct_count += 1
        feature_line = test_features.readline()
        label_line = test_labels.readline()
    test_features.close()
    test_labels.close()
    #print 'Pecent correctly classified: ' + str((correct_count / (diff_count + correct_count)))
    return str((correct_count / (diff_count + correct_count)))

examples = get_training_examples('training-features.txt', 'training-labels.txt')
for i in range(30):
    classifier = BaggingClassifier(1, examples)
    result = ensemble_classify_file('test-features.txt', 'test-labels.txt', classifier)
    print result
    classifier.add_learner()



