import random
import pandas as pd
import functools
import glob
import os
import numpy as np
import h5py
import platform
from loguru import logger
from typing import List, Tuple
from valtes_radartoolbox.data.framefragment import FrameFragment
from valtes_radartoolbox.data.framewindow import FrameWindow, FrameWindowLabel
from valtes_radartoolbox.data.frame import Frame
from valtes_radartoolbox.data.rangedoppler import RangeDoppler
from valtes_radartoolbox.config.ti_config import ConfigFile

import matplotlib.pyplot as plt
from valtes_radartoolbox.mmwave.dataloader import DCA1000

def read_hdf5(filename, config=None, label_set=None):
    try:
        df = pd.read_hdf(filename)
    except:
        return
    if label_set:
        df = df.merge(
            pd.read_csv(label_set).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'])), 
            on='timestamp', 
            how='left'
            )

    for index, row in df.iterrows():
        timestamp = row["timestamp"]
        frame = row.get("frame", None) # TLVs
        image = row.get("image", None) # raw jpeg
        label = row.get("label", None)
        adc   = row.get("mmwave", None)
        frame = Frame(timestamp, frame, image, label, config)
        if adc is not None:
            frame.set_radar(adc)
        yield frame

def read_directory(dirname, config=None, read_func=read_hdf5, label_set=None):
    files = get_directory_files(dirname)
    for filename in files:
        for frame in read_hdf5(filename, config, label_set):
            yield frame

def get_directory_files(dirname):
    return sorted(glob.glob(os.path.join(dirname , "*.h5"), recursive=False))

class BaseDataset(object):
    def __init__(self, _type="TLV"):
        if platform.system() == "Windows":
            self._basedir = os.path.join("G:", os.sep, "Shared drives", "Data", _type)
        if platform.system() == "Linux":
            self._basedir = os.path.join("/mnt/g/Shared drives/Data/", _type)
        else:
            # self._basedir = os.path.join("smb://127.0.0.1/Shared drives/Data/TLV/", _type)
            pass

    def get_directory_files(self, dirname, extension="*.h5"):
        files =  sorted(glob.glob(os.path.join(dirname , extension), recursive=False))
        if len(files) == 0:
            logger.error("Empty dataset {}, no {} data files", dirname, extension)
            logger.error(os.listdir(dirname))
        return files

class ValtesDataset(BaseDataset):
    def __init__(self, data_dir: str, label_set=None, debug=False, type_="TLV"):
        super().__init__(type_)
        self._dir = os.path.join(self._basedir, data_dir)
        self._debug = debug
        if not os.path.exists(self._dir):
            logger.error("Not a directory: '{}'", self._dir)
            return
        self._read_configs()
        self._read_data()
        self._read_labels(label_set)

    def _read_data(self):
        self._data_files = self.get_directory_files(self._dir, "*.h5")

    def _read_configs(self):
        self._cfg_files = self.get_directory_files(self._dir, "*.cfg")
        
    def _read_labels(self, label_set):
        self._label_set = label_set
        if label_set:
            self._labels_path = os.path.join(self._dir, label_set)
        else:
            self._labels_path = None

    def __iter__(self):
        for frame in read_directory(self._dir, config=self.config, label_set=self._labels_path):
            yield frame

    @property
    def config(self):
        if len(self._cfg_files):
            try:
                cfg = ConfigFile()
                cfg.parseCfg(self._cfg_files[0])
                return cfg
            except Exception as e:
                print(e, self._cfg_files)
                return None
        logger.warning("No config file available for dataset {}", self)
    
    def read_frames(self):
        frames = list(read_directory(self._dir, config=self.config, label_set=self._labels_path))
        return frames

    @functools.cached_property
    def frames(self):
        return self.read_frames()

    def random_frame(self):
        rd_int = random.randint(0, len(self.frames))
        return self.frames[rd_int]


class TLVDataset(ValtesDataset):
    def __init__(self, datadir, label_set=None):
        super().__init__(datadir, label_set)

class RawDataIterator:
    def __init__(self, data):
        self._data = data
        self._idx = 0

    def __iter__(self):
        return self
 
    def __next__(self):
        if self._idx >= len(self._data):
            raise StopIteration
        f = Frame()
        f.set_radar(self._data[self._idx,:,:,:])
        self._idx+=1
        return f

class RawDataset(ValtesDataset):
    def __init__(self, datadir, label_set=None):
        super().__init__(datadir, type_="DCA1000")
    
    def __len__(self):
        return len(self._all_data)
    
    def __iter__(self) -> Frame:
        return RawDataIterator(self._all_data)
    
    def _read_data(self):
        self._data_files = self.get_directory_files(self._dir, "*.bin")
        config = self.config
        sensor = config.sensor()
        for f in self._data_files:
            # the int16 is the matlab scripts in TI documentation
            # openradar had uint16 bit this created overflows in ADC samples (but good point clouds)
            adc_data = np.fromfile(f, dtype=np.int16)
#            adc_data = adc_data - (adc_data >= 2**15) * 2**16

            print(sensor.num_chirps_per_frame, sensor.num_rx_antennas, sensor.num_adc_samples)
            est_frame_size  =sensor.num_chirps_per_frame * sensor.num_rx_antennas * sensor.num_adc_samples
            print(est_frame_size)

            try:
                adc_data = adc_data.reshape((-1,
                                            2* #x2 because complex
                                            sensor.num_chirps_per_frame *
                                            sensor.num_rx_antennas *
                                            sensor.num_adc_samples))
            except Exception as e:
                print(sensor.num_chirps_per_frame, sensor.num_rx_antennas, sensor.num_adc_samples)
                est_frame_size  =sensor.num_chirps_per_frame * sensor.num_rx_antennas * sensor.num_adc_samples
                print(est_frame_size)
                print(adc_data.shape[0] / float(est_frame_size))
                print(e)
                raise
            print(adc_data.shape)
            all_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=sensor.num_chirps_per_frame,
                               num_rx=sensor.num_rx_antennas, num_samples=sensor.num_adc_samples)

            self._all_data = all_data

class RawDataIterator2:
    def __init__(self, h5):
        self._h5 = h5
        self._idx = 0

    def __iter__(self):
        return self
 
    def __next__(self):
        print(len(self._h5))
        f = Frame()
        img = self._h5["rgb"][self._idx]
        rad = self._h5["radar"][self._idx]
        if img is not None:
            f.set_image(img)
        f.set_radar(rad)
        self._idx+=1
        return f

class RawDataset2(ValtesDataset):
    def __init__(self, datadir):
        super().__init__(datadir, type_="DCA1000")

    def _load_single_h5(self, filename):
        h5 = h5py.File(filename, 'r')
        streams_available = list(h5.keys())
        dataset_len = len(h5[streams_available[0]])
        return h5, streams_available, dataset_len

    def __len__(self):
        h5, streams_available, dataset_len = self._load_single_h5(self._data_files[0])
        return dataset_len
    
    def __iter__(self) -> Frame:
        h5, streams_available, dataset_len = self._load_single_h5(self._data_files[0])
        return RawDataIterator2(h5)
    
    def __pieter__(self) -> Frame:
        h5 = {}
        f = Frame()
        idx = 0
        img = h5["rgb"][idx]
        rad = h5["radar"][idx]
        f.set_image(img)
        f.set_radar(rad)
        yield f

class TrainDataset(ValtesDataset):
    """Dataset used for training predictive algorithms with PyTorch. 
    If used in combination with a DataLoader the dataset will return the original 
    windows and range-doppler heatmaps for training."""
    def __init__(self, datadir, label_set=None, window_size=15, outlier_treshold=10000):
        if platform.system() == "Windows":
            print("Windows")
            super().__init__(os.path.join("G:/Shared drives/Data/TLV/", datadir), label_set)

        if platform.system() == "Linux":
            print("Linux")
            super().__init__(os.path.join("/mnt/g/Shared drives/Data/TLV/", datadir), label_set)

        
        super().__init__(os.path.join("smb://127.0.0.1/Shared drives/Data/TLV/", datadir), label_set)

        print("Check")
        super().read_frames()
        self.window_size = window_size
        self.transform = None
        self.outliers_mask = self.create_outliers_mask(outlier_treshold)
    
    def __iter__(self) -> FrameWindow:
        for window in self.create_frame_windows():
            yield window

    def __len__(self) -> int:
        return len(self.frame_windows)
    
    def __getitem__(self, idx: int) -> Tuple[RangeDoppler, FrameWindow, FrameWindowLabel]:
        """Method that is called under the hood by the PyTorch data loader. 
        Transforms heatmaps if a transform function is set.

        Args:
            idx (int): idx to select window

        Returns:
            Range-doppler heatmap, FrameWindow and FrameWindowLabel
        """
        window = self.frame_windows[idx]   
        heatmaps = np.array([frame.output.range_doppler_heatmap for frame in window.frames]).astype(float)
        label = window.label.label

        if self.transform:
            heatmaps = self.transform(heatmaps)
        
        return heatmaps, window, label

    def set_transform(self, transform):
        self.transform = transform

    @property
    def frames(self) -> np.array:
        return np.array(super().frames)[self.outliers_mask]

    @functools.cached_property
    def frame_windows(self) -> List[FrameWindow]:
        return self.create_frame_windows()
    
    @property
    def frame_fragments(self) -> List[FrameFragment]:
        return self.create_frame_fragments()

    def create_outliers_mask(self, threshold: int) -> np.array:
        """Makes a mask to filter out outliers. Outliers are defined as empty range-doppler heatmaps or range-doppler heatmaps with a
        maximu value above a certain threshold

        Returns:
            np.array: A mask with booleans specifying which frames are outliers
        """
        outliers_idxs = []
        for idx, frame in enumerate(super().frames):
            if np.array(frame.output.range_doppler_heatmap).shape[0] == 0:
                outliers_idxs.append(idx)
            elif np.max(np.array(frame.output.range_doppler_heatmap)) > threshold:
                outliers_idxs.append(idx)
            
        logger.info(f"max_idx = {idx}")
        
        mask = np.ones(idx + 1, dtype=bool)
        mask[outliers_idxs] = False
        
        return mask

    def get_frame_labels_df(self) -> pd.DataFrame:
        """Load frame labels and contextual information derived from a .csv file

        Returns:
            pd.DataFrame: A dataframe containing context information and labels from all frames in the dataset
        """
        if self._labels_path:
            return (
                pd.read_csv(self._labels_path)
                # Remove outliers
                [self.outliers_mask]
                # Reset indices to align with frames
                .reset_index(drop=True).reset_index().rename(columns={'index': 'frame_num'})
                # Identify label switches
                .assign(label_previous_frame=lambda x: x['label'].shift())
                .assign(label_next_frame=lambda x: x['label'].shift(-1))
                .assign(label_switch=lambda x: x['label'] != x['label_previous_frame'])
                .assign(label_group_num=lambda x: x['label_switch'].cumsum())
                )
        else:
            logger.error("No label set provided, can not load labels")
    
    @property
    def frame_labels(self) -> pd.DataFrame:
        return self.get_frame_labels_df()['label']
    
    @property
    def frame_window_labels(self) -> List[FrameWindowLabel]:
        return [frame_window.label.label for frame_window in self.frame_windows]

    def get_frame_fragment_labels_df(self) -> pd.DataFrame:
        """Load FrameFragment labels and contextual information derived from a the frame labels

        Returns:
            pd.DataFrame: A dataframe containing context information and labels from all FrameFragments in the dataset
        """
        return (
            self.get_frame_labels_df()
            .groupby(['label', 'label_group_num'])
            .agg({'timestamp': ['min', 'max', 'count'], 'frame_num': ['min', 'max'], 'label_next_frame':'last', 'label_previous_frame': 'first'})
            .assign(duration = lambda x: (pd.to_datetime(x[('timestamp', 'max')]) - pd.to_datetime(x[('timestamp', 'min')])).dt.seconds)
            .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1))
            .rename(columns={'frame_num_min': 'first_frame',
                             'frame_num_max': 'last_frame',
                             'timestamp_count': 'number_of_frames', 
                             'duration_': 'total_duration', 
                             'label': 'window_state_label', 
                             'label_previous_frame_first': 'previous_state', 
                             'label_next_frame_last': 'next_state'})
            .reset_index()
            .sort_values('first_frame')
            # Drop last and first window, because that is a movement window including movement from standing to lying down
            .iloc[1:-1]
            # .loc[lambda x: x['label'] != 99]
            .reset_index()
        )
    
    def create_frame_fragments(self) -> List[FrameFragment]:
        """Create FrameFragments from all Frames in the dataset

        Returns:
            List[FrameFragment]: A list of FrameFragments
        """
        transitions_label_dict = {0: 'no transition', 1: 'back to side', 2: 'side to back', 3: 'front to side', 4: 'side to front'}
        state_label_dict = {0: 'moving', 1: 'back', 2: 'side', 3: 'front', 99: 'standing'}

        fragments = []
        for i, row in self.get_frame_fragment_labels_df().loc[lambda x: x['number_of_frames'] >= self.window_size].iterrows():
            # Create single windows for all transitions
            # TODO: Make sure that transitions are always labeled as zero, or add a label-name to the df_labels
            if row['label'] == 99 or row['previous_state'] == 99 or row['next_state'] == 99:
                pass
            elif row['label'] == 0: 
                # Get the frame number of the first frame in the state
                first_frame = row['first_frame']
                last_frame = row['last_frame']
                # Create a fragment
                fragments.append(FrameFragment(
                    label=FrameWindowLabel(state_label=row['label'], transition_label=f"{state_label_dict[row['previous_state']]} to {state_label_dict[row['next_state']]}"),
                    first_frame_idx=first_frame, 
                    last_frame_idx=last_frame,
                    frames=self.frames[first_frame:last_frame + 1]
                    ))
            else:
                # Get the frame duration and number for the first and last frame
                first_frame = row['first_frame']
                last_frame = row['last_frame']
                # Select and create the window
                fragments.append(FrameFragment(
                    label=FrameWindowLabel(state_label=row['label'], transition_label=transitions_label_dict[0]),
                    first_frame_idx=first_frame,
                    last_frame_idx=last_frame,
                    frames=self.frames[first_frame:last_frame + 1]
                    ))
        return fragments


    def create_frame_windows(self) -> List[FrameWindow]:
        """Extract a FrameWindow from all FrameFragments
        
        Returns:
            List[FrameFragment]: A list of FrameWindows in the dataset"""
        windows = []
        for fragment in self.frame_fragments:
            windows.append(fragment.extract_window(window_size=self.window_size))
        
        return windows
    
    def plot_labels_mean_doppler(self, label_dict=None, plot_windows=False):
        rd_hmps = [frame.output.range_doppler_heatmap for frame in np.array(self.frames)[self.outliers_mask]]

        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        ax.scatter(np.arange(0, len(np.array(self.frames)[self.outliers_mask])), self.frame_labels, label="label")
        for i, row in self.get_frame_labels_df().loc[lambda x: x['label_switch'] == True].iterrows():
            ax.vlines(i, ymin=row['label'], ymax=row['label_previous_frame'])

        if plot_windows:
            for window in self.frame_windows:
                ax.fill_betweenx([0, 2], window._first_frame, window._last_frame, color='gray', alpha=0.2, label='Extracted Windows')
            

        ax2 = ax.twinx()
        ax2.plot(np.mean(rd_hmps, axis=(1, 2)), color='red', linestyle='--', label="Mean doppler")
        
        y_tick_labels = [0, 1, 2, 3]
        if label_dict:
            y_tick_labels = [label_dict[y_tick_label] for y_tick_label in y_tick_labels]
        ax.set_yticks([0, 1, 2, 3], labels=y_tick_labels)
        ax.set_xlabel('Frame number')
        ax.set_ylabel('Label')
        ax.set_ylim((0, 3))

        # Combine the legend from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        # Add the legend from the second axis to the first axis
        plt.legend(lines[:2] + lines2, labels[:2] + labels2, loc='upper right')

def list_datasets(type_="TLV"):
    bd = BaseDataset(type_)
    datasets = os.listdir(bd._basedir)
    return datasets