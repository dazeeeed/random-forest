import numpy as np
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
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _entropy(self, y):
        """Calculate entropy"""
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) if 0 < p < 1 else 0 for p in proportions])

    def _create_split(self, X, threshold):
        """Find the indices in X that meet threshold criteria and collapse in one dimension"""
        left_idx = np.argwhere(X <= threshold).flatten()
        right_idx = np.argwhere(X > threshold).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, threshold):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._create_split(X, threshold)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = (n_left / n ) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        """Create best split at current node"""
        # TODO: Implement selekcja progowa w połaczeniu z ruletka here
        split = {"score": -1, "feature": None, "threshold": None}

        for feature in features:
            X_i = X[:, feature]
            thresholds = np.unique(X_i)
            for thr in thresholds:
                score = self._information_gain(X_i, y, thr)

                if score > split["score"]:
                    split["score"] = score
                    split["feature"] = feature
                    split["threshold"] = thr

        return split["feature"], split["threshold"]

    def _is_finished(self, depth):
        """Evaluate criteria if building process is finished"""
        if depth >= self.max_depth or self.n_class_labels == 1 or self.n_samples < self.min_samples_split:
            return True
        return False

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
    def __init__(self, n_estimators=100, max_depth=100, min_sample_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self._trees = [DecisionTreeClassifier(self.max_depth, min_samples_split=self.min_sample_split)
                       for _ in range(self.n_estimators)]

    def _draw_bootstrap(self, X, y):
        """Draw random indices and return bootstrap data based on them."""
        bootstrap_idx = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
        X_bootstrap, y_bootstrap = X[bootstrap_idx], y[bootstrap_idx]
        return X_bootstrap, y_bootstrap

    def fit(self, X, y):
        for i, tree in enumerate(self._trees):
            if i+1 % 5 == 0:
                print("Training tree number {}".format(i+1))
            X_bootstrap, y_bootstrap = self._draw_bootstrap(X, y)
            tree.fit(X_bootstrap, y_bootstrap)

    def predict(self, X):
        predictions_trees = np.empty(shape=(self.n_estimators, X.shape[0]), dtype=int)
        for i, tree in enumerate(self._trees):
            predictions_trees[i, :] = np.array(tree.predict(X))

        predictions_forest = [np.bincount(predictions_trees[:, i]).argmax() for i in range(X.shape[0])]
        return predictions_forest
