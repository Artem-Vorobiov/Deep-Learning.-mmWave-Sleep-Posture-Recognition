import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# TRANSITION_LABEL_DICT = {0: 'no transition', 1: 'back to side', 2: 'side to back', 3: 'front to side', 4: 'side to front'}
TRANSITION_LABEL_DICT = {0: 'back', 1: 'back to side', 2: 'side to back', 3: 'front to side', 4: 'side to front', 5: "side", 6: "front"}
TRANSITION_LABEL_DICT_INV = {v: k for k, v in TRANSITION_LABEL_DICT.items()}

class FrameFragment(object):
    """A subset of frames capturing a specific pose or transition sequence."""
    def __init__(self, label, first_frame_idx, last_frame_idx, frames):
        self._label = label
        self._first_frame = first_frame_idx
        self._last_frame = last_frame_idx
        self._frames = frames
    
    @property
    def frames(self):
        return self._frames
    
    @property
    def frame_idxs(self):
        return self._first_frame, self._last_frame

    @property
    def label(self):
        return self._label
    
    @property
    def length(self):
        return self._last_frame - self._first_frame + 1
    
    def extract_window(self, window_size):
        if self._label.state_label == 0:
            windows = sliding_window_view([frame.output.range_doppler_heatmap for frame in self.frames], window_size, axis=0)
            windows_mean = windows.mean(axis=(1, 2, 3))
            # Select the window (list of frames) with the highest mean range-doppler
            selected_window_idx = np.argmax(windows_mean)
            selected_window = self.frames[selected_window_idx: selected_window_idx+window_size]
        
        else:
            selected_window_idx = np.random.randint(0, self.length - window_size + 1)
            selected_window = self.frames[selected_window_idx: selected_window_idx+window_size]
            # Select and create the window

        return FrameWindow(
                label=self._label,
                first_frame_idx=selected_window_idx, 
                last_frame_idx=selected_window_idx + window_size, 
                frames=selected_window)

class FrameWindow(object):
    """A set number of frames derived from a FrameFragment and contains a subset of a specific pose or transition fragment."""
    def __init__(self, label, first_frame_idx, last_frame_idx, frames):
        self._label = label
        self._first_frame = first_frame_idx
        self._last_frame = last_frame_idx
        self._frames = frames

    @property
    def frames(self):
        return self._frames

    @property
    def label(self):
        return self._label

class FrameWindowLabel(object):
    
    def __init__(self, state_label, transition_label):
        self._state_label = state_label
        self._transition_label = TRANSITION_LABEL_DICT_INV[transition_label]
        self._transition_label_str = transition_label

    @property
    def state_label(self):
        return self._state_label
    
    @property
    def label(self):
        return self._transition_label
    
    @property
    def transition_label_str(self):
        return self._transition_label_str
    
    def __str__(self) -> str:
        return self._transition_label_str

