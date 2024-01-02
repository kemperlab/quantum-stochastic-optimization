from .problem import QSOProblem
from .tight_binding import TightBindingProblem, TightBindingParameters
from .feature_selection import FeatureSelectionProblem, FeatureSelectionParameters

ProblemParameters = TightBindingParameters | FeatureSelectionParameters
