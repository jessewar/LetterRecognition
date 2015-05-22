import decision_tree
import random
from collections import Counter

class BaggingClassifier:
      def __init__(self, num_learners, examples):
            self.num_learners = num_learners
            self.examples = examples
            self.learners = []
            # Initialize set classifiers
            for i in range(num_learners):
                  self.learners.append(BaseLearner(self.examples))

      # Classifies an example by majority vote of all classifiers
      def classify_example(self, example, correct_label):
            counter = Counter()
            for learner in self.learners:
                  classification = learner.classify_example(example, correct_label)
                  counter[classification] += 1
            return max(counter, key=counter.get)

      # Adds a learner to the set of learners used in classification
      def add_learner(self):
            self.learners.append(BaseLearner(self.examples))

class BaseLearner:
      def __init__(self, examples):
            sample = []
            for i in range(len(examples)):
                  random_index = random.randint(0, len(examples) - 1)
                  sample.append(examples[random_index])
            self.root = decision_tree.construct_tree(sample, range(0, 16), 0.01)
            # tuples = []
            # for data_point in sample:
            #       tuples.append((data_point[:16], data_point[16]))
            # self.root = jack_tree.buildTree(tuples, range(0, 16), 1)
      
      def classify_example(self, example, correct_label):
            return self.classify_example_helper(example, correct_label, self.root)

      def classify_example_helper(self, example, correct_label, curr_root):
          if curr_root.classification is not None:
                return curr_root.classification
          # if curr_root.letter is not None:
          #       return curr_root.letter
          else:
                split_attribute = curr_root.attribute
                attribute_value = example[split_attribute]
                return self.classify_example_helper(example, correct_label, curr_root.children[attribute_value])

