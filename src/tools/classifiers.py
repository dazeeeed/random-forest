# --------------------------------------------------
# Created By  : Krzysztof Palmi, Łukasz Sawicki 
# Created Date: 04.06.2022
# Class:        Machine Learning, 2022 summer
# --------------------------------------------------

import numpy as np
from numpy.random import choice
from IPython.display import clear_output


class Node:
    """Helper class with information about feature, threshold value and children"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 100, min_samples_split: int = 2, min_samples_leaf: int = 1,
                 criterion: str = 'entropy', threshold: float = 0.4, split_method: str = 'roulette'):
        """
        :param max_depth: The maximum depth of the tree. Default 100.
        :param min_samples_split: The minimum number of samples required to split an internal node. Default 2.
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node. Default 1.
        :param criterion: {“gini”, “entropy”}. The function to measure the quality of a split. Default 'entropy'.
        :param threshold: Threshold used to reject possible splits at a leaf node. Default 0.4.
        :param split_method: {"roulette", "classic"}. Method to select split at the leaf node.
            Classic will split at the place with the biggest score, roulette will choose place with roulette method
            considering all possible splits with score bigger than threshold.
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.threshold = threshold
        self.method = split_method
        self.root = None

    def _criterion(self, y):
        """Calculate the quality of a split, based on selected function in parameter 'criterion'"""
        if self.criterion == 'entropy':
            return self._entropy(y)
        elif self.criterion == 'gini':
            return self._gini(y)

    def _entropy(self, y):
        """Calculate entropy"""
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) if 0 < p < 1 else 0 for p in proportions])

    def _gini(self, y):
        """Calculate gini index"""
        proportions = np.bincount(y) / len(y)
        return 2 * (1 - np.sum([p**2 for p in proportions]))

    def _create_split(self, X, threshold):
        """Find the indices in X that meet threshold criteria and collapse in one dimension"""
        left_idx = np.argwhere(X <= threshold).flatten()
        right_idx = np.argwhere(X > threshold).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, threshold):
        parent_loss = self._criterion(y)
        left_idx, right_idx = self._create_split(X, threshold)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = (n_left / n) * self._criterion(y[left_idx]) + (n_right / n) * self._criterion(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        if self.method == 'classic':
            return self._best_split_classic(X, y, features)
        elif self.method == 'roulette':
            return self._best_split_roulette(X, y, features)

    def _best_split_classic(self, X, y, features):
        """Create best split at current node with classic selection"""
        split = {"score": -1, "feature": None, "threshold": None}

        for feature in features:
            X_i = X[:, feature]
            thresholds = np.unique(X_i)
            for thr in thresholds:
                # # check if leaf size is satisfied with this threshold
                # if not self._is_leaf_enough(X, thr):
                #     continue

                score = self._information_gain(X_i, y, thr)
                if score > split["score"]:
                    split["score"] = score
                    split["feature"] = feature
                    split["threshold"] = thr

        return split["feature"], split["threshold"]

    def _best_split_roulette(self, X, y, features):
        """Create best split at current node with roulette selection"""
        population = list()

        # return when max_score < threshold
        best_split = {"score": -1, "feature": None, "threshold": None}

        for feature in features:
            X_i = X[:, feature]
            thresholds = np.unique(X_i)
            for thr in thresholds:
                # # check if leaf size is satisfied with this threshold
                # if not self._is_leaf_enough(X, thr):
                #     continue

                score = self._information_gain(X_i, y, thr)
                population.append({'score': score, 'feature': feature, 'threshold': thr})

                if score > best_split["score"]:
                    best_split["score"] = score
                    best_split["feature"] = feature
                    best_split["threshold"] = thr

        max_score = max(population, key=lambda d: d['score'])['score']

        if max_score <= self.threshold:
            return best_split['feature'], best_split['threshold']

        # reject scores below the threshold
        population = list(filter(lambda d: d['score'] > self.threshold, population))
        population = sorted(population, key=lambda d: d['score'], reverse=True)

        # TODO druga część wybierania progowego, nie rozumem go
        # jakiś kod

        # roulette
        prob_sum = np.sum([d['score'] for d in population])
        split = choice(population, p=[d['score'] / prob_sum for d in population])

        return split['feature'], split['threshold']

    def _is_finished(self, depth):
        """Evaluate criteria if building process is finished"""
        return depth >= self.max_depth or self.n_class_labels == 1 or self.n_samples < self.min_samples_split

    def _is_leaf_enough(self, X, threshold):
        """Evaluate criteria for new created leaf"""
        left_idx, right_idx = self._create_split(X, threshold)
        return X[right_idx, :].shape[0] >= self.min_samples_leaf and X[left_idx, :].shape[0] >= self.min_samples_leaf

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # Stopping criteria
        if self._is_finished(depth):
            most_common_label = np.argmax(np.bincount(y))
            return Node(value=most_common_label)

        # Select random features without replacement and create best split based on them
        random_features = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, random_features)

        # Recursive growing of children
        left_idx, right_idx = self._create_split(X[:, best_feature], best_threshold)

        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_threshold, left_child, right_child)

    def _traverse_tree(self, x, node):
        """Traverse through tree looking for a prediction for sample x"""
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)


class RandomForestClassifier:
    def __init__(self, n_estimators: int = 100, max_depth: int = 100, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, criterion: str = 'entropy', threshold: float = 0.4,
                 split_method: str = 'roulette'):
        """
        :param n_estimators: The number of trees in the forest. Default 100.
        :param max_depth: The maximum depth of the tree. Default 100.
        :param min_samples_split: The minimum number of samples required to split an internal node. Default 2.
        :param min_samples_leaf: The minimum number of samples required to be at a leaf node. Default 1.
        :param criterion: {“gini”, “entropy”}. The function to measure the quality of a split. Default 'entropy'.
        :param threshold: Threshold used to reject possible splits at a leaf node. Default 0.4.
        :param split_method: {"roulette", "classic"}. Method to select split at the leaf node.
            Classic will split at the place with the biggest score, roulette will choose place with roulette method
            considering all possible splits with score bigger than threshold.
        """

        assert criterion == 'entropy' or criterion == 'gini', \
            f"An invalid value for parameter 'criterion' was given: {criterion}. Available are: {{“gini”, “entropy”}}."
        assert split_method == 'classic' or split_method == 'roulette', f"Type correct method"

        assert 0.0 < threshold < 1.0, f"Parameter 'threshold' must be between 0.0 and 1.0."

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.threshold = threshold
        self.method = split_method
        self._trees = [DecisionTreeClassifier(
            self.max_depth, min_samples_split=self.min_samples_split, criterion=self.criterion, threshold=self.threshold)
            for _ in range(self.n_estimators)]

    def _draw_bootstrap(self, X, y):
        """Draw random indices and return bootstrap data based on them."""
        bootstrap_idx = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
        X_bootstrap, y_bootstrap = X[bootstrap_idx], y[bootstrap_idx]
        return X_bootstrap, y_bootstrap

    def fit(self, X, y):
        for i, tree in enumerate(self._trees):
            if (i + 1) % 5 == 0:
                print(f"Training tree number {i + 1}")
            X_bootstrap, y_bootstrap = self._draw_bootstrap(X, y)
            tree.fit(X_bootstrap, y_bootstrap)

    def predict(self, X):
        predictions_trees = np.empty(shape=(self.n_estimators, X.shape[0]), dtype=int)
        for i, tree in enumerate(self._trees):
            predictions_trees[i, :] = np.array(tree.predict(X))

        predictions_forest = [np.bincount(predictions_trees[:, i]).argmax() for i in range(X.shape[0])]
        return predictions_forest
