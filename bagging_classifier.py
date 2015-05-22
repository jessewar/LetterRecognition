import decision_tree

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
            for learner in learners:
                  classification = learner.classify_example(example, correct_label)
                  counter[classification] += 1
            return max(counter, counter.get)

class BaseLearner:
      def __init__(self, examples, p_value=1):
            self.p_value = p_value
            sample = []
            for i in range(len(examples)):
                  random_index = random.randint(0, len(examples) - 1)
                  sample.append(examples[random_index])
            self.root = decision_tree.construct_tree(sample, range(0, 16), self.p_value)
      
      def classify_example(self, example, correct_label):
          if self.root.classification is not None:
                return root.classification
          else:
                split_attribute = root.attribute
                attribute_value = example[split_attribute]
                return classify_example(root.children[attribute_value], example, correct_label)

