class Node:

    def __init__(self, attribute=None, classification=None, examples=[]):
        self.attribute = attribute
        self.classification = classification
        self.children = {}
        self.examples = examples
        self.hit_count = 0
        self.num_correct = 0

    def __str__(self, level = 0):
        result = '\t' * level
        result += 'attribute: ' + str(self.attribute) + ', '
        #result += 'examples: ' + str(self.examples) + ', '
        result += 'classification: ' + str(self.classification) + '\n'
        for attribute_value in self.children:
            child = self.children[attribute_value]
            result += child.__str__(level + 1)
        return result
