import numpy as np
import cv2

from valtes_radartoolbox.data.image import Image
from valtes_radartoolbox.parsing_utils.parseFrame import parseStandardFrame
from valtes_radartoolbox.data.output import Output

class FrameLabel(object):
    
    def __init__(self, label):
        self._label = label
    
    @property
    def label(self):
        return self._label
    
    def __str__(self) -> str:
        return self._label

class Frame(object):
    def __init__(self, timestamp=None, tlvs=None, image=None, label=None, config=None):
        self._timestamp = timestamp
        self._tlvs = tlvs
        self._image: str = image # raw jpeg
        self._rgb_image: np.ndarray = None
        self._label: FrameLabel = FrameLabel(label)
        self._config = config
        self._output = None
        self._radar = None
        
    def __str__(self):
        if self._output:
            return str(self._timestamp) + " " + str(self.output) + " has_image=" + str(self._image is not None)
        if self._radar is not None:
            return str(self._timestamp) + " " + str(self._radar.shape) + " has_image=" + str(self._image is not None)
        return str(self._timestamp)

    @property
    def frame(self):
        return self._tlvs
    
    @property
    def image(self) -> Image:
        if self._image is not None:
            if self._rgb_image is None:
                # would be nicer to have this in Image class
                jpeg_image = np.frombuffer(self._image, dtype=np.uint8)
                image = cv2.imdecode(jpeg_image, cv2.IMREAD_COLOR)
                self._rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self._rgb_image = Image(self._rgb_image)
            return self._rgb_image
        return Image(None)
    
    def jpg(self):
        if self._image is not None:
            jpeg_image = np.frombuffer(self._image, dtype=np.uint8)
            return jpeg_image
    
    def _parse_tlvs(self):
        """
            internal function
        """
        outputDict, framedata = parseStandardFrame(self._tlvs, self._config)
        output = Output(outputDict)
        return output

    @property
    def output(self):
        if self._output is None:
            self._output = self._parse_tlvs()

        return self._output
    
    @output.setter
    def output(self, new_output):
        self._output = new_output
    
    def output2(self):
        outputDict, framedata = parseStandardFrame(self._tlvs, self._config)
        return outputDict
    
    @property
    def timestamp(self):
        return self._timestamp
    
    @property
    def label(self):
        return self._label
    
    def set_image(self, image):
        self._image = image
        self._rgb_image = Image(image)

    def set_radar(self, radar):
        self._radar = radar 
