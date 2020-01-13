import numpy as np


class ConfusionMatrix:
    """Streaming interface to allow for any source of predictions. Initialize it, count predictions one by one, then print confusion matrix and intersection-union score"""

    def __init__(self, number_of_labels=2):
        self.number_of_labels = number_of_labels
        self.confusion_matrix = np.zeros(shape=(self.number_of_labels, self.number_of_labels))

    @staticmethod
    def create_from_matrix(confusion_matrix):
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
        matrix = ConfusionMatrix(confusion_matrix.shape[0])
        matrix.confusion_matrix = confusion_matrix
        return matrix

    def count_predicted(self, ground_truth, predicted, number_of_added_elements=1):
        self.confusion_matrix[ground_truth][predicted] += number_of_added_elements

    def count_predicted_batch(self, ground_truth_vec, predicted):
        assert np.max(predicted) < self.number_of_labels
        for i in range(ground_truth_vec.shape[0]):
            self.confusion_matrix[ground_truth_vec[i], predicted[i]] += 1

    def count_predicted_batch_hard(self, ground_truth_vec, predicted):  # added
        for i in range(ground_truth_vec.shape[0]):
            self.confusion_matrix[ground_truth_vec[i], predicted[i]] += 1

    def get_count(self, ground_truth, predicted):
        """labels are integers from 0 to number_of_labels-1"""
        return self.confusion_matrix[ground_truth][predicted]

    def get_confusion_matrix(self):
        """returns list of lists of integers; use it as result[ground_truth][predicted]
            to know how many samples of class ground_truth were reported as class predicted"""
        return self.confusion_matrix

    def get_intersection_union_per_class(self):
        """returns list of 64-bit floats"""
        matrix_diagonal = [self.confusion_matrix[i][i] for i in range(self.number_of_labels)]
        errors_summed_by_row = [0] * self.number_of_labels
        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                if row != column:
                    errors_summed_by_row[row] += self.confusion_matrix[row][column]
        errors_summed_by_column = [0] * self.number_of_labels
        for column in range(self.number_of_labels):
            for row in range(self.number_of_labels):
                if row != column:
                    errors_summed_by_column[column] += self.confusion_matrix[row][column]

        divisor = [0] * self.number_of_labels
        for i in range(self.number_of_labels):
            divisor[i] = matrix_diagonal[i] + errors_summed_by_row[i] + errors_summed_by_column[i]
            if matrix_diagonal[i] == 0:
                divisor[i] = 1

        return [float(matrix_diagonal[i]) / divisor[i] for i in range(self.number_of_labels)]

    def get_overall_accuracy(self):
        """returns 64-bit float"""
        confusion_matrix = self.confusion_matrix
        matrix_diagonal = 0
        all_values = 0
        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                all_values += confusion_matrix[row][column]
                if row == column:
                    matrix_diagonal += confusion_matrix[row][column]
        if all_values == 0:
            all_values = 1
        return float(matrix_diagonal) / all_values

    def no_interest_indexes(self, ignored_indexes):
        self.ignored_indexes = np.array(ignored_indexes)

    def get_average_intersection_union(self,):
        values = self.get_intersection_union_per_class()
        class_seen = ((self.confusion_matrix.sum(1) + self.confusion_matrix.sum(0)) != 0).sum()
        if class_seen == 0:
            return 0
        return sum(values) / class_seen

    def get_mean_class_accuracy(self):  # added
        re = 0
        label_presents = 0
        for i in range(self.number_of_labels):
            total_gt = np.sum(self.confusion_matrix[i, :])
            if total_gt:
                label_presents += 1
                re = re + self.confusion_matrix[i][i] / max(1, total_gt)
        if label_presents == 0:
            return 0
        return re / label_presents

    def count_gt(self, ground_truth):
        return self.confusion_matrix[ground_truth, :].sum()
