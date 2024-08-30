from valtes_radartoolbox.data.framewindow import FrameWindow


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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