import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PointCloud(pd.DataFrame):    
    # This class variable tells Pandas the name of the attributes
    # that are to be ported over to derivative DataFrames.  There
    # is a method named `__finalize__` that grabs these attributes
    # and assigns them to newly created `PointCloud`
    _metadata = ['_xyz_vec', '_xyz_vecn']

    def __init__(self, *args, **kwargs):
        self._xyz_vec = kwargs.pop('xyzVec', None)
        self._xyz_vecn = kwargs.pop('xyzVecN', None)
        super(PointCloud, self).__init__(*args, **kwargs)
          
    @property
    def _constructor(self):
        """This is the key to letting Pandas know how to keep
        derivative `PointCloud` the same type as yours.  It should
        be enough to return the name of the Class.  However, in
        some cases, `__finalize__` is not called and `_xyz_vec` is
        not carried over.  We can fix that by constructing a callable
        that makes sure to call `__finalize__` every time."""
        def _c(*args, **kwargs):
            return PointCloud(self._xyz_vec, self._xyz_vecn, *args, **kwargs).__finalize__(self)
        return _c

    def cluster(self, method=None):
        pass

    def plot_3d(self):
        fig = plt.figure()
        nice = fig.add_subplot(111, projection='3d')  # Updated to use add_subplot for compatibility

        nice.set_zlim3d(bottom=-5, top=5)
        nice.set_ylim(bottom=0, top=10)
        nice.set_xlim(left=-4, right=4)
        nice.set_xlabel('X Label')
        nice.set_ylabel('Y Label')
        nice.set_zlabel('Z Label')

        nice.scatter(self._xyz_vec[0], self._xyz_vec[1], self._xyz_vec[2], c='r', marker='o', s=2)
        plt.show()

    def plot_xy(self):
        fig, axes = plt.subplots(1, 2)
        xyzVec = self._xyz_vec
        xyzVecN = self._xyz_vecn
        xyzVec = xyzVec[:, (np.abs(xyzVec[2]) < 1.5)]
        xyzVecN = xyzVecN[:, (np.abs(xyzVecN[2]) < 1.5)]
        axes[0].set_ylim(bottom=0, top=10)
        axes[0].set_ylabel('Range')
        axes[0].set_xlim(left=-4, right=4)
        axes[0].set_xlabel('Azimuth')
        axes[0].grid(visible =True)

        axes[1].set_ylim(bottom=0, top=10)
        axes[1].set_xlim(left=-4, right=4)
        axes[1].set_xlabel('Azimuth (m)')
        axes[1].grid(visible =True)
        axes[0].scatter(xyzVec[0], xyzVec[1], c='r', marker='o', s=3)
        axes[1].scatter(xyzVecN[0], xyzVecN[1], c='b', marker='o', s=3)

    def plot_xz(self):
        xyzVec = self._xyz_vec
        xyzVecN = self._xyz_vecn
    
        fig, axes = plt.subplots(1, 2)

        axes[0].set_ylim(bottom=-5, top=5)
        axes[0].set_ylabel('Elevation')
        axes[0].set_xlim(left=-4, right=4)
        axes[0].set_xlabel('Azimuth')
        axes[0].grid(visible =True)

        axes[1].set_ylim(bottom=-5, top=5)
        axes[1].set_xlim(left=-4, right=4)
        axes[1].set_xlabel('Azimuth')
        axes[1].grid(visible =True)


        axes[0].scatter(xyzVec[0], xyzVec[2], c='r', marker='o', s=3)
        axes[1].scatter(xyzVecN[0], xyzVecN[2], c='b', marker='o', s=3)

    def project(self):
        pass


if __name__ == "__main__":
    pc = PointCloud({ "a" : [1,2,3,4], "b" : [3,4,5,6]})
    print(pc.describe())

