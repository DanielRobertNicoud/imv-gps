import numpy as np
import os

def blender_full_picture(picture_name, mean_inputs, mean, uncertainty, X=None, y=None):
    """
        picture_name             name of the picture (str)
        mean_inputs              basepoints of the mean vectors (n, 3)
        mean                     mean vectors (n, 3)
        uncertainty_inputs       basepoints of uncertainties (n, 3)
        uncertainty              uncertainties (n,)
        X, y                     observations (m, 3), can be None
    """
    out_folder = os.path.join("blender-data", "outputs")
    
    np.savetxt(os.path.join(out_folder, f"{picture_name}__mean.csv"), np.hstack([mean_inputs, mean]), delimiter=",")
    np.savetxt(os.path.join(out_folder, f"{picture_name}__std.csv"), uncertainty, delimiter=",")
    if X is not None:
        np.savetxt(os.path.join(out_folder, f"{picture_name}__observations.csv"), np.hstack([X, y]), delimiter=",")