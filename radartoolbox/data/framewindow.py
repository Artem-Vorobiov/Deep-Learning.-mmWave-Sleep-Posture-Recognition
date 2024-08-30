from typing import List
from valtes_radartoolbox.data.frame import Frame

TRANSITION_LABEL_DICT = {0: 'no transition', 1: 'back to side', 2: 'side to back', 3: 'front to side', 4: 'side to front'}
TRANSITION_LABEL_DICT_INV = {v: k for k, v in TRANSITION_LABEL_DICT.items()}

class FrameWindowLabel(object):
    """A Label belonging to a FrameWindow. Contains a state_label that describes the state of the window (e.g. 'lying on the side' or 'transitioning 
    from one position to the other') and a transition label that describves a particular transition (e.g. 'side -> back' or 'front -> side', can also be
    'no transition')
    """
    
    def __init__(self, state_label, transition_label):
        self._state_label = state_label
        self._transition_label = TRANSITION_LABEL_DICT_INV[transition_label]
        self._transition_label_str = transition_label

    @property
    def state_label(self) -> int:
        return self._state_label
    
    @property
    def label(self) -> int:
        return self._transition_label
    
    @property
    def transition_label_str(self) -> str:
        return self._transition_label_str
    
    def __str__(self) -> str:
        return self._transition_label_str

class FrameWindow(object):
    """A set number of frames derived from a FrameFragment and contains a subset of a specific pose or transition fragment."""
    def __init__(self, label, first_frame_idx, last_frame_idx, frames):
        self._label = label
        self._first_frame = first_frame_idx
        self._last_frame = last_frame_idx
        self._frames = frames

    @property
    def frames(self) -> List[Frame]:
        return self._frames

    @property
    def label(self) -> FrameWindowLabel:
        return self._label
