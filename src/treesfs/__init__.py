"""treesfs: decision trees (CART) and random forests from scratch in NumPy."""

from treesfs.config import Config, load_config
from treesfs.criteria import CRITERIA, entropy, gini, impurity_decrease, squared_error
from treesfs.export import export_graphviz, export_text, render_dot
from treesfs.metrics import accuracy, mae, mse, r2_score
from treesfs.models.forest import (
    RandomForestClassifier,
    RandomForestRegressor,
    forest_from_config,
)
from treesfs.models.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    TreeArrays,
    tree_from_config,
)
from treesfs.permutation_importance import PermutationImportanceResult, permutation_importance
from treesfs.preprocessing import OneHotEncoder, build_design_matrix

__version__ = "0.2.0"

__all__ = [
    "CRITERIA",
    "Config",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "OneHotEncoder",
    "PermutationImportanceResult",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "TreeArrays",
    "__version__",
    "accuracy",
    "build_design_matrix",
    "entropy",
    "export_graphviz",
    "export_text",
    "forest_from_config",
    "gini",
    "impurity_decrease",
    "load_config",
    "mae",
    "mse",
    "permutation_importance",
    "r2_score",
    "render_dot",
    "squared_error",
    "tree_from_config",
]
