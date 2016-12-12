# Network class called by main, with train, test functions

from datasets import get_dataset

class Network(object):
    def __init__(self, params):
        self.params = params
        self.dataset = get_dataset(params)
        self.dataset.get_files_labels_list()
        # self.dataset.setup_graph()

    def train(self):
        pass

