"""

"""
import pickle


class NAS_DNN_Benchmark:
    def __init__(self, dataset_loader, architecture_list):
        self.dataset_loader = dataset_loader
        self.architecture_list = architecture_list
        self.precomputed_metrics = {}

    def precompute_metrics(self):
        for architecture in self.architecture_list:
            accuracy, interpretability, flops = architecture.evaluate_all_objectives(self.dataset_loader)
            self.precomputed_metrics[str(architecture.hidden_sizes)] = {
                'accuracy': accuracy,
                'interpretability': interpretability,
                'flops': flops
            }

    def save_precomputed_metrics(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.precomputed_metrics, f)

    def load_precomputed_metrics(self, filename):
        with open(filename, 'rb') as f:
            self.precomputed_metrics = pickle.load(f)
