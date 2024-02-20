import numpy as np
from tqdm import tqdm
import jax
from sklearn.gaussian_process import GaussianProcessRegressor
from aux_functions import flatten_coord, unflatten_coord

class SphereVectorGP:
    """
    Wrapper around scikit-learn GP regressor.
    
    Inputs (X, y) are (n, 2) jax arrays in spherical coordinates.
    """
    
    def __init__(self, kernel, **kwargs):
        self.gp = GaussianProcessRegressor(kernel=kernel, **kwargs)
        self.log_marginal_likelihood_value_ = None
        
    def fit(self, X, y):
        X = np.atleast_2d(X)
        assert X.ndim == 2 and X.shape[1] == 2
        self.gp.fit(flatten_coord(X)[:, None], flatten_coord(y))
        self.log_marginal_likelihood_value_ = self.gp.log_marginal_likelihood_value_
    
    def predict(self, X, return_std=False, return_cov=False, verbose=False):
        X = np.atleast_2d(X)
        assert X.ndim == 2 and X.shape[1] == 2
        if not (return_std or return_cov):
            y = unflatten_coord(self.gp.predict(
                flatten_coord(X)[:, None], return_std=False, return_cov=False
            ))
            return y
        
        if return_std and return_cov:
            raise RuntimeError("Only one of return_std and return_cov can be set to True.")
        
        if return_std:
            y_and_std = [self.__single_predict_with_std(x) for x in (tqdm(X) if verbose else X)]
            y = np.stack([y for y, _ in y_and_std])
            std = np.stack([std for _, std in y_and_std])
            return y, std
            
        raise NotImplementedError("return_cov")
    
    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__, self.gp.kernel_)
    
    def sample_y(self, X, n_samples=1, random_state=0):
        y_samples = self.gp.sample_y(flatten_coord(X)[:, None], n_samples, random_state)
        return unflatten_coord(y_samples, extra_dims=[n_samples])
    
    def __single_predict_with_std(self, x):
        # note: std is a matrix!
        y, cov = self.gp.predict(x[:, None], return_std=False, return_cov=True)
        return y, cov