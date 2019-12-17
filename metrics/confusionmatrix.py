import numpy as np


class ConfusionMatrix:
    """Streaming interface to allow for any source of predictions. Initialize it, count predictions one by one, then print confusion matrix and intersection-union score"""

    def __init__(self, number_of_labels=2):
        self.number_of_labels = number_of_labels
        self.confusion_matrix = np.zeros(shape=(self.number_of_labels, self.number_of_labels))
        self._modified_confusion_matrix = None
        self.ignored_indexes = []  # DEFAULT IT IS SET TO NONE

    def set_modified_matrix(self, modified_confusion_matrix):
        self._modified_confusion_matrix = modified_confusion_matrix

    def count_predicted(self, ground_truth, predicted, number_of_added_elements=1):
        self.confusion_matrix[ground_truth][predicted] += number_of_added_elements

    def count_predicted_batch(self, ground_truth_vec, predicted):  # added
        # Added code ---------------------------------------------------------------
        gt_vec_copy = np.zeros((ground_truth_vec.shape[0], self.confusion_matrix.shape[1]))
        gt_vec_copy[:, :ground_truth_vec.shape[1]] = ground_truth_vec
        for i in range(ground_truth_vec.shape[0]):
            if len(ground_truth_vec[i, :]) != len(self.confusion_matrix[:, predicted[i]]):
                self.confusion_matrix[:, predicted[i]] += gt_vec_copy[i, :]
            else:
                # Added code ---------------------------------------------------------------
                self.confusion_matrix[:, predicted[i]] += ground_truth_vec[i, :]

    def count_predicted_batch_hard(self, ground_truth_vec, predicted):  # added
        for i in range(ground_truth_vec.shape[0]):
            self.confusion_matrix[ground_truth_vec[i], predicted[i]] += 1

    """labels are integers from 0 to number_of_labels-1"""

    def get_count(self, ground_truth, predicted):
        return self.confusion_matrix[ground_truth][predicted]

    """returns list of lists of integers; use it as result[ground_truth][predicted]
     to know how many samples of class ground_truth were reported as class predicted"""

    def get_confusion_matrix(self):
        return self.confusion_matrix

    """returns list of 64-bit floats"""

    def get_intersection_union_per_class(self, alpha=None, modified=False):

        if alpha is None:
            alpha = .5

        if modified:
            confusion_matrix = self._modified_confusion_matrix
        else:
            confusion_matrix = self.confusion_matrix

        matrix_diagonal = [confusion_matrix[i][i] for i in range(self.number_of_labels)]
        errors_summed_by_row = [0] * self.number_of_labels
        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                if row != column:
                    errors_summed_by_row[row] += confusion_matrix[row][column]
        errors_summed_by_column = [0] * self.number_of_labels
        for column in range(self.number_of_labels):
            for row in range(self.number_of_labels):
                if row != column:
                    errors_summed_by_column[column] += confusion_matrix[row][column]

        divisor = [0] * self.number_of_labels
        for i in range(self.number_of_labels):
            iou_denom = 2 * ((1 - alpha) * errors_summed_by_row[i] + alpha * errors_summed_by_column[i])
            divisor[i] = matrix_diagonal[i] + iou_denom
            if matrix_diagonal[i] == 0:
                divisor[i] = 1

        return [float(matrix_diagonal[i]) / divisor[i] for i in range(self.number_of_labels)]
    """returns 64-bit float"""

    def get_overall_accuracy(self, modified=False):
        if modified:
            confusion_matrix = self._modified_confusion_matrix
        else:
            confusion_matrix = self.confusion_matrix
        matrix_diagonal = 0
        all_values = 0
        for row in range(self.number_of_labels):
            for column in range(self.number_of_labels):
                if (modified) and (row in self.ignored_indexes):  # IF MODIFIED, SKIP WHEN GROUND TRUTH IS IN IGNORED_INDEXES
                    continue
                all_values += confusion_matrix[row][column]
                if row == column:
                    matrix_diagonal += confusion_matrix[row][column]
        if all_values == 0:
            all_values = 1
        return float(matrix_diagonal) / all_values

    def no_interest_indexes(self, ignored_indexes):
        self.ignored_indexes = np.array(ignored_indexes)

    def get_average_intersection_union(self, alpha=None, modified=False, debug=False):
        if not modified:
            values = self.get_intersection_union_per_class(alpha=alpha)
            class_seen = ((self.confusion_matrix.sum(1)+self.confusion_matrix.sum(0)) != 0).sum()
            return sum(values) / class_seen
        else:
            values = np.array(self.get_intersection_union_per_class(alpha=alpha, modified=modified))
            if hasattr(self, "ignored_indexes"):
                values[self.ignored_indexes] = 0
            mask = values > 0
            class_seen = ((self.confusion_matrix.sum(1)+self.confusion_matrix.sum(0)) != 0)
            class_seen = class_seen[mask].sum()
            if debug:
                return sum(values) / class_seen, values
            return sum(values) / class_seen

    def get_mean_class_accuracy(self):  # added
        re = 0
        label_presents = 0
        for i in range(self.number_of_labels):
            total_gt = np.sum(self.confusion_matrix[i, :])
            if total_gt:
                label_presents += 1
                re = re + self.confusion_matrix[i][i] / max(1, total_gt)
        return re/label_presents

    def count_gt(self, ground_truth):
        return self.confusion_matrix[ground_truth, :].sum()
