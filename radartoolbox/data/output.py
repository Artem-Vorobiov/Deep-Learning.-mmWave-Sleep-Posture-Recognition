from loguru import logger
import numpy as np
from valtes_radartoolbox.data.rangedoppler import RangeDoppler
import pandas as pd

class TrackData(np.ndarray):
    def __new__(cls, trackdata):
        obj = np.asarray(trackdata).view(cls)
        return obj

    def to_pandas(self):
        if len(self):
            return pd.DataFrame(self, columns=["target_id", "x", "y", "z", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "G", "conf", "None1", "None2", "None3", "None4"])
        return pd.DataFrame()
        
class PointCloud(np.ndarray):
    def __new__(cls, pointcloud, pointmeta=None):
        obj = np.asarray(pointcloud).view(cls)
        return obj

    def __init__(self, pointcloud, pointmeta=None):
        self._pointmeta = pointmeta

    def to_pandas(self):
        df1 = pd.DataFrame(self, columns=["X", "Y", "Z", "Doppler", "SNR", "Noise", "Track index"])
        df2 = pd.DataFrame.from_records(self._pointmeta)
        return pd.concat([df1, df2], axis=1)

class TargetIndexes(np.ndarray):
    def __new__(cls, trackindexes):
        obj = np.asarray(trackindexes).view(cls)
        return obj

    def to_pandas(self):
        df1 = pd.DataFrame(self, columns=["target_id"])
        return df1

                            
class Output(object):
    def __init__(self, output_dict):
        
        self._output = output_dict

        self._error = self._output.pop("error", None)
        self._frame_num = self._output.pop("frameNum", "x")
        self._height_data = self._output.pop("heightData", [])
        self._pointcloud = self._output.pop("pointCloud", [])
        self._num_detected_heights = self._output.pop("numDetectedHeights", 0)
        self._num_detected_points = self._output.pop("numDetectedPoints", 0)
        self._num_detected_tracks = self._output.pop("numDetectedTracks", 0)
        self._pointmeta = self._output.pop("pointMeta", [])
        self._presence_indication = self._output.pop("presenceIndication", None)
        self._trackdata = self._output.pop("trackData", [])
        self._track_indexes = self._output.pop("trackIndexes", [])
        self._range_doppler_heatmap = self._output.pop("rangeDopplerHeatmap", [])
        self._range_profile = self._output.pop("rangeProfile", [])
        self._stats = self._output.pop("stats", [])
        self._temperature_stats = self._output.pop("temperatureStats", None)        
        
        rest = self._output.keys()
        if len(rest):
            logger.error("Keys left in output dict {}", rest)

    def __str__(self):
        return "Frame #" + str(self._frame_num)

    def as_dict(self):
        return {
            "error" : self._error,
            "frameNum" : self._frame_num,
            "heightData" : self._height_data,
            "pointCloud" : self._pointcloud,
            "numDetectedPoints" : self._num_detected_points,
            "numDetectedTracks" : self._num_detected_tracks,
            "trackData" : self._trackdata
        }

    @property
    def error(self):
        return self._error

    @property
    def frame_num(self):
        return self._frame_num

    @property
    def num_detected_heights(self):
        return self._num_detected_points

    @property
    def num_detected_points(self):
        return self._num_detected_points
    
    @property
    def num_detected_tracks(self):
        return self._num_detected_tracks

    @property
    def pointcloud(self) -> PointCloud:
        return PointCloud(self._pointcloud, self._pointmeta)

    @property
    def range_doppler_heatmap(self) -> RangeDoppler:
        return self._range_doppler_heatmap
    
    @range_doppler_heatmap.setter
    def range_doppler_heatmap(self, new_range_doppler_heatmap):
        self._range_doppler_heatmap = new_range_doppler_heatmap

    @property
    def trackdata(self) -> TrackData:
        return TrackData(self._trackdata)
    
    @property
    def trackindexes(self) -> TargetIndexes:
        return TargetIndexes(self._track_indexes)
