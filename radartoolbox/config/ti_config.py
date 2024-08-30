from loguru import logger
from typing import Any
import numpy as np
from ..gui_common import *
import tabulate
from typing import List

NUM_HEART_RATES_FOR_MEDIAN = 10
NUM_FRAMES_PER_VITALS_PACKET = 15
MAX_TX = 3
MAX_RX = 4

def antenna_enabled(mask) -> List[int]:
    en = []
    for channel in range(4): # MAX RX
        if mask & (0x01 << channel):
            en.append(channel+1)
    return en

class Board(object):
    """ Physical settings belonging to the hardware board """
    
class CFAR(object):
    ## This is the category CFAR DPU
    def __init__(self):
        pass


class GTRACK(object):
    """
        The DPU for GTRACK, the configuration parameters can be subdivided in:
        - scenery
        - gating (physical contraints)
        - allocation (new tracks)
        - state (tracking state transition)
        - acceleration
    """
    def __init__(self):
        # the bounding box coordinates in which the tracker is allowed to operate
        self._boundary_box = None
        # the bounding box in which targets are becoming static for longer periods of time
        self._static_boundary_box = None
        # the bounding box is the origin of tracks
        self._presence_boundary_box = None



class Cfg(object):
    pass

class BoundaryBox(Cfg):
    pass

class PresenceBoundaryBox(BoundaryBox):
    pass

class StaticBoundaryBox(BoundaryBox):
    pass

class FovCfg(Cfg):
    def parse(self, args):
        """
        This command defines the sensor azimuth and elevation FOVs. The FOV values in this command define 
        the total angular extent observed at one side of the sensor. In other words, a value of 𝜃 in this command 
        configures the corresponding FOV to lie in the closed interval [−𝜃, 𝜃]. Targets outside of this angular 
        FOV interval are not detected in the people counting demo
        """
        self._subFrameIdx = int(args[1])
        self._azimuthFoV = float(args[2])
        self._elevationFoV = float(args[3])

class ChannelCfg(Cfg):
    def describe(self):
        table = [
            ["RX channel enabled", antenna_enabled(self._rx_channel_en)],
            ["TX channel enabled", antenna_enabled(self._tx_channel_en)],
            ["Cascading", self._cascading]
        ]
        print(tabulate.tabulate(table))

    @property
    def num_rx_antennas(self) -> int:
        return len(antenna_enabled(self._rx_channel_en))

    @property
    def num_tx_antennas(self) -> int:
        return len(antenna_enabled(self._tx_channel_en))

    def parse(self, args):
        self._rx_channel_en = int(args[1])
        self._tx_channel_en = int(args[2])
        self._cascading = int(args[3]) # N/A

class AdcCfg(Cfg):
    def describe(self):
        pass

class CalibDcRangeSig(Cfg):
    """
        Antenna coupling signature dominates the range bins close to 
        the radar. These are the bins in the range FFT output located 
        around DC.
        When this feature is enabled, the signature is estimated during 
        the first N chirps, and then it is subtracted during the 
        subsequent chirps.
        During the estimation period the specified bins (defined as 
        [negativeBinIdx, positiveBinIdx]) around DC are accumulated 
        and averaged. It is assumed that no objects are present in the 
        vicinity of the radar at that time
        """
    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._enabled = int(args[2])
        self._negativeBinIdx = int(args[3])
        self._positiveBinIdx = int(args[4])
        # Value of 256 means first 256 chirps (after command is issued and feature is enabled) will be used for 
        # collecting (averaging) DC signature in the bins specified above. From 257th chirp, the collected DC 
        # signature will be removed from every chirp.
        self._numAvg = int(args[5])

class DynamicRangeAngleCfg(Cfg):
    """
    For the dynamic scenes, the range-angle heatmap estimation step uses Capon beamforming in the 2D 
    range-azimuth domain for the 3D wall-mount chain and 3D range-azimuth-elevation domain for the 3D 
    ceil-mount chain. The configuration parameters are detailed below.
    """
    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._angleSearchStep = float(args[2])
        self._rangeAngleDiagonalLoading = float(args[3])
        self._rangeAngleEstMethod = int(args[4])
        self._dopplerEstMethod = int(args[5])

class DynamicRACfarCfg(Cfg):
    """
    Part of the detection chain
    """
    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._cfarDiscardLeftRange = int(args[2])
        self._cfarDiscardRightRange = int(args[3])
        self._cfarDiscardLeftAngle = int(args[4])
        self._cfarDiscardRightAngle = int(args[5])
        self._refWinSizeRange = int(args[6])
        self._refWinSizeAngle = int(args[7])
        self._guardWinSizeRange = int(args[8])
        self._guardWinSizeAngle = int(args[9])
        self._rangeThre = float(args[10])
        self._angleThre = float(args[11])
        self._sidelobeThre = float(args[12])
        self._enable2ndPass = int(args[13])
        self._dynamicFlag = int(args[14])

class StaticRACfarCfg(Cfg):
    """
    Part of the detection chain
    """
    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._cfarDiscardLeftRange = int(args[2])
        self._cfarDiscardRightRange = int(args[3])
        self._cfarDiscardLeftAngle = int(args[4])
        self._cfarDiscardRightAngle = int(args[5])
        self._refWinSizeRange = int(args[6])
        self._refWinSizeAngle = int(args[7])
        self._guardWinSizeRange = int(args[8])
        self._guardWinSizeAngle = int(args[9])
        self._rangeThre = float(args[10])
        self._angleThre = float(args[11])
        self._sidelobeThre = float(args[12])
        self._enable2ndPass = int(args[13])
        self._dynamicFlag = int(args[14])

class AdcBufCfg(Cfg):
    def describe(self):
        pass

    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._adcOutputFmt = int(args[2])
        self._sampleSwap = int(args[3]) # only option 1 is supported
        self._chanInterleave = int(args[4]) # only option 1 is supported
        self._chirpThreshold = int(args[5]) # 
        assert(self._sampleSwap==1)
        assert(self._chanInterleave==1)

class AntGeometry(Cfg):
    """
        Antenna geometry parameters antGeometry0 and antGeometry1 define the virtual antennas' physical 
        location index (0, -1, -2, …) in the azimuth and elevation domains, respectively
    """
    def parse(self, args):
        self._virtAntIdx0 = int(args[1])
        self._virtAntIdx1 = int(args[2])
        self._virtAntIdx2 = int(args[3])
        self._virtAntIdx3 = int(args[4])
        self._virtAntIdx4 = int(args[5])
        self._virtAntIdx5 = int(args[6])
        self._virtAntIdx6 = int(args[7])
        self._virtAntIdx7 = int(args[8])
        self._virtAntIdx8 = int(args[9])
        self._virtAntIdx9 = int(args[10])
        self._virtAntIdx10 = int(args[11])
        self._virtAntIdx11 = int(args[12])

class AntPhaseRot(Cfg):
    """
        This parameter defines the phase rotation introduced in the board design. Each field should be set to 1 if 
        no rotation exists and -1 if there is a phase rotation in the corresponding antenna element
    """
    def parse(self, args):
        self._virtAntIdx0 = int(args[1])
        self._virtAntIdx1 = int(args[2])
        self._virtAntIdx2 = int(args[3])
        self._virtAntIdx3 = int(args[4])
        self._virtAntIdx4 = int(args[5])
        self._virtAntIdx5 = int(args[6])
        self._virtAntIdx6 = int(args[7])
        self._virtAntIdx7 = int(args[8])
        self._virtAntIdx8 = int(args[9])
        self._virtAntIdx9 = int(args[10])
        self._virtAntIdx10 = int(args[11])
        self._virtAntIdx11 = int(args[12])

class StateParam(Cfg):
    def parse(self, args):
        self._det2actThre = int(args[1])
        self._det2freeThre = int(args[2])
        self._active2freeThre = int(args[3])
        self._static2freeThre = int(args[4])
        self._exit2freeThre = int(args[5])
        self._sleep2freeThre = int(args[6])

class BpmCfg(Cfg):
    def describe(self):
        pass
    
    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._enabled = int(args[2])
        self._chirp0Idx = int(args[3])
        self._chirp1Idx = int(args[4])

class LvdsStreamCfg(Cfg):
    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._enableHeader = int(args[2])
        self._dataFmt = int(args[3])
        self._enableSW = int(args[4])

class CfarCfg:
    def describe(self):
        pass

class CfarFovCfg:
    def describe(self):
        pass

class MaxAcceleration(Cfg):
    def parse(self, args):
        self._max_x = float(args[1])
        self._max_y = float(args[2])
        self._max_z = float(args[3])

class MultiObjBeamForming(Cfg):
    """
    Multi Object Beamforming config message to datapath. 
    This feature allows radar to separate reflections from multiple 
    objects originating from the same range/Doppler detection.
    The procedure searches for the second peak after locating the 
    highest peak in Azimuth FFT. If the second peak is greater than 
    the specified threshold, the second object with the same range
    /Doppler is appended to the list of detected objects. The 
    threshold is proportional to the height of the highest peak
    """
    def parse(self, args):
        self._subFrameIdx = int(args[1]) # should be set to -1
        self._enabled = int(args[2])
        self._threshold = float(args[3])

class FineMotionCfg(Cfg):
    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._fineMotionProcEnabled = int(args[2])
        self._fineMotionObservationTime = float(args[3])
        self._fineMotionProcCycle = int(args[4])
        self._fineMotionDopplerThrIdx = int(args[5])

class ChirpCfg(Cfg):
    def __eq__(self, other):
        return self._freq_slope == other._freq_slope and \
               self._idle_time == self._idle_time
    
    def __str__(self):
        return "C" + str(self._freq_slope)
    
    def describe(self):
        table = [
            ["start_idx", self._start_idx ],
            ["end_idx", self._end_idx],
            ["profile_id", self._profile_id ],
            ["start_freq", self._start_freq ],
            ["freq_slope", self._freq_slope],
            ["idle_time", self._idle_time ],
            ["adc_start_time", self._adc_start_time],
            ["tx_ant_mask", antenna_enabled(self._tx_ant_mask)]
        ]
        print(tabulate.tabulate(table))

    def doc(self):
        print("idle_time: The time between the end of previous chirp and start of next chirp")
        print("profile_id: ")
        print("tx_ant_mask: Individual chirps should have either only one distinct Tx antenna enabled (MIMO) or same TX antennas should be enabled for all chirps")
    
    def parse(self, args):
        self._start_idx = int(args[1])
        self._end_idx = int(args[2])
        self._profile_id = int(args[3])
        self._start_freq = float(args[4])
        self._freq_slope = float(args[5])
        self._idle_time = float(args[6])
        self._adc_start_time = float(args[7])
        self._tx_ant_mask = int(args[8])
        assert(self._start_idx == self._end_idx)
        assert(self._start_idx in [0,1,2])
        assert(self._end_idx in [0,1,2])

class ClutterRemoval(Cfg):
    """
    Static clutter removal algorithm implemented by subtracting 
    from the samples the mean value of the input samples to the 
    2D-FFT """
    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._enabled = int(args[2])

    def enabled(self) -> bool:
        return bool(self._enabled)
    
class CompRangeBiasAndRxChanPhase(Cfg):
    def parse(self, args):
        self._re00 = float(args[1])
        self._im00 = int(args[2])
        self._re01 = int(args[3])
        self._im01 = int(args[4])
        self._re02 = int(args[5])
        self._im02 = int(args[6])
        self._re03 = int(args[7])
        self._im03 = int(args[8])
        self._re04 = int(args[9])
        self._im04 = int(args[10])
        self._re05 = int(args[11])
        self._im05 = int(args[12])
        self._re06 = int(args[13])
        self._im06 = int(args[14])
        self._re07 = int(args[15])
        self._im07 = int(args[16])
        self._re08 = int(args[17])
        self._im08 = int(args[18])
        self._re09 = int(args[19])
        self._im09 = int(args[20])
        self._re10 = int(args[21])
        self._im10 = int(args[22])
     #   self._re11 = int(args[23])
     #   self._im11 = int(args[24])
     #   self._re12 = int(args[25])
     #   self._im12 = int(args[26])

class Dynamic2DAngleCfg(Cfg):
    """When the range-angle heatmap estimation method 1 is used (i.e., rangeAngleEstMethod = 1), for each 
    point detected in the range-azimuth domain, a Capon beamforming algorithm is applied to generate the 
    elevation spectrum, which will be used to estimate the elevation angle of the detected point
    """
    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._elevSearchStep = float(args[2])
        self._angleDiagonalLoading = float(args[3])
        self._maxNpeak2Search = int(args[4])
        self.peakExpSamples = int(args[5])
        self._elevOnly = int(args[6])
        self._sidelobeThre = float(args[7])
        self._peakExpRelThre = float(args[8])
        self._peakExpSNRThre = float(args[9])

class StaticRangeAngleCfg(Cfg):
    """
    The range-angle heatmap estimation step uses Bartlett beamforming in the range-azimuth-elevation 
    domain for the static scenes
    """
    def parse(self, args):
        self._subFrameIdx = int(args[1])
        self._staticProcEnabled = int(args[2])
        self._staticAzimStepDeciFactor = int(args[3])
        self._staticElevStepDeciFactor = int(args[4])

class FrameCfg(Cfg):
    def describe(self):
        table = [
          #  ["ChirpStartIndex", self._chirp_start_index],
        #    ["ChirpEndIndex", self._chirp_end_index],
            ["NumberOfLoops", self._number_of_loops]
        ]
        print(tabulate.tabulate(table))

    def doc(self):
        # number of loops should be a multiple of 4
        print("NumberOfLoops: the number of chirps per frame")

    @property
    def num_loops(self):
        return self._number_of_loops
    
    def parse(self, args):
        if (len(args) < 4):
            print ("Error: frameCfg had fewer arguments than expected")
            return False
        self._chirp_start_index = int(args[1])
        self._chirp_end_index = int(args[2])
        self._number_of_loops = int(args[3])
        self._number_of_frames = int(args[4])
        self._frame_periocidity = float(args[5])
        self._trigger_select = int(args[6])
        self._frame_trigger_delay = float(args[7])
    
        return True
     
class GuiMonitorCfg:
    def __getitem__(self, __name: str) -> Any:
        return self.__dict__["_" + __name]
    
    def __setitem__(self, __name: str, __value) -> Any:
        self.__dict__["_" + __name] = __value
    
    def describe(self):
        table = [
            ["detectedObjects",  str(self._detectedObjects)],
            ["logMagRange ", str(self._logMagRange)],
            ["noiseProfile ", str(self._noiseProfile)],
            ["rangeAzimuthElevationHeatMap", str(self._rangeAzimuthElevationHeatMap)],
            ["rangeDopplerHeatMap ", self._rangeDopplerHeatMap],
            ["statsInfo ", str(self._statsInfo)],
        ]
        print(tabulate.tabulate(table))

class ProfileCfg(Cfg):
    def __eq__(self, other):
        return self._start_freq == other._start_freq and \
                self._freq_slope_const == other._freq_slope_const
        
    def __getitem__(self, __name: str) -> Any:
        return self.__dict__["_" + __name]

    def __sub__(self, other):
        a = self.table()
        b = other.table()

        result = []
        for x1, x2 in zip(a,b):
            if x1[1] != x2[1]:
                result.append([x1[0], x1[1], x2[1]])
        if len(result):
            return tabulate.tabulate(result)

    def describe(self):
       print(tabulate.tabulate(self.table()))
       print("THUS: ADC Sampling Time", self._num_adc_samples / self._dig_out_sample_rate)
       print("THUS: Sweep bandwidth", self._freq_slope_const * (self._num_adc_samples / self._dig_out_sample_rate) )
       print()

    def parse(self, args):
        self._profile_id = int(args[1]) # 0-3
        self._start_freq = float(args[2]) # eg 77, 61.38
        self._idle_time = float(args[3]) # eg 7, 7.15
        self._adc_start_time = float(args[4]) # eg 7, 7.34
        self._ramp_end_time = float(args[5]) # eg 58, 216.15
        self._tx_out_power = float(args[6]) # always 0
        self._tx_phase_shifter = float(args[7]) # always 0
        self._freq_slope_const = float(args[8]) # 68, 16.83
        self._tx_start_time = float(args[9]) # 1, 2.92
        self._num_adc_samples = float(args[10]) # 256, 224
        self._dig_out_sample_rate = float(args[11]) / 1e3 # 5500 (directly convert from k to M)
        self._hpf_corner_freq1 = int(args[12]) # 0-3
        self._hpf_corner_freq2 = int(args[13]) # 0-3
        self._rx_gain = args[14]
    
    @property
    def dig_out_sample_rate(self) -> float:
        """
            returns the sample rate in Ms/s
        """
        return self._dig_out_sample_rate
    
    @property
    def num_adc_samples(self) -> int:
        return int(self._num_adc_samples)
    
    @property
    def num_range_bins(self) -> int:
        # in the chip it is rounded to 2, however the datacube size
        # depends on the number of adc samples actually returned
       # return int(self._num_adc_samples)

        # the nr range bins is ceil(log2(nrAdcSamples))
        numAdcSamplesRoundTo2 = 1
        while self._num_adc_samples > numAdcSamplesRoundTo2:
            numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
        return numAdcSamplesRoundTo2

    def table(self):
        table = [
            ["Profile", "#" + str(self._profile_id)],
            ["Start frequency ", str(self._start_freq) + " GHz"],
            ["Frequency Slope ", str(self._freq_slope_const) + " MHz/us"],
            ["ADC sampling frequency", str(self._dig_out_sample_rate) + " Ms/s"],
            ["Samples per chirp ", self._num_adc_samples],
            ["Idle Time ", str(self._idle_time) + " us"],
            ["ADC Valid Start Time", self._adc_start_time],
            ["Ramp End Time", str(self._ramp_end_time) + " us"],
            ["TX Start Time", self._tx_start_time],
           # ["TX power", self._tx_out_power], # tx power is not used
            #["TX phase", self._tx_phase_shifer]
        ]
        return table

class RadarSS(object):
    """ The radar subsystem"""
    def __init__(self):
        self._channel : ChannelCfg = None
        self._chirps : List[ChirpCfg] = []
        self._profiles: List[ProfileCfg] = []
        self._frames : List[FrameCfg] = []
    
    def __eq__(self, other):
        ret = True
        
        for p1, p2 in zip(self._profiles, other._profiles):
            ret &= (p1 == p2)
        for c1, c2 in zip(self._chirps, other._chirps):
            ret &= (c1 == c2)
            
        return ret
    
    def __sub__(self, other):
        p = self.profile - other.profile
        return p
    
    def add_chirp(self, chirp: ChirpCfg):
        self._chirps.append(chirp)

    def add_frame(self, frame: FrameCfg):
        self._frames.append(frame)
    
    def add_profile(self, profile: ProfileCfg):
        self._profiles.append(profile)

    def describe(self):
        self._channel.describe()
        for p in self._frames:
            p.describe()
        for p in self._profiles:
            p.describe()
        for p in self._chirps:
            p.describe()

    @property
    def doppler_bins(self):
        """Create doppler bins centralized around 0"""
        return np.multiply(
            np.arange(-self.num_doppler_bins // 2 + 0.5,
                    self.num_doppler_bins // 2 + 0.5, 
                    1), 
                    self.doppler_resolution)
    
    @property
    def doppler_resolution(self):
        """Calculate the doppler resolution for the given radar configuration.

        Args:
            start_freq_const (int): Frequency chirp starting point.
            ramp_end_time (int): Frequency chirp end point.
            idle_time_const (int): Idle time between chirps.
            band_width (float): Radar config bandwidth.
            num_loops_per_frame (int): The number of loops in each frame.
            num_tx_antennas (int): The number of transmitting antennas (tx) on the radar.

        Returns:
            doppler_resolution (float): The doppler resolution for the given radar configuration.

        """
        #band_width, start_freq_const=77, ramp_end_time=62, idle_time_const=100, num_loops_per_frame=128,
        #                   num_tx_antennas=3
        profile = self._profiles[0]

        start_freq_const = profile._start_freq
        ramp_end_time = profile._ramp_end_time
        idle_time_const = profile._idle_time
        num_tx_antennas = self.num_tx_antennas
    
        light_speed_meter_per_sec = 299792458

        chirp_interval = (ramp_end_time + idle_time_const) * 1e-6
        doppler_resolution = light_speed_meter_per_sec / ( 2 * start_freq_const * 1e9 * chirp_interval * self.num_doppler_bins * num_tx_antennas)
        return doppler_resolution

    def get_range_max(self, visibility_factor=0.8):
        """
            visibility_factor = 0.8 # TI specific, take SNR into account based on software version

            Returns the maximum range in meters of this config
        """
        if_max = self.profile.dig_out_sample_rate
        c = 299792458 # speed of light (m/s)
        
        beat_freq = visibility_factor * if_max
        S = self.profile._freq_slope_const * 1e6  # slope of the transmitted chirp (MHz/s)
        range_max = (beat_freq * c) / ( 2 * S)

        return range_max

    @property
    def num_adc_samples(self) -> int:
        return self._profiles[0].num_adc_samples
        
    @property
    def num_chirps_per_frame(self) -> int:
        # this might be wrong if the chirps are not all saved in the cfg file
        return int(len(self._chirps) * self._frames[0].num_loops)

    @property
    def num_doppler_bins(self):
        # Typically the number of doppler bins is the same as the number of chirps per tx antenna
        return int(self.num_chirps_per_frame // self.num_tx_antennas)

    @property
    def num_range_bins(self):
        return self._profiles[0].num_range_bins
    
    @property
    def num_rx_antennas(self) -> int:
        return self._channel.num_rx_antennas

    @property
    def num_tx_antennas(self) -> int:
        # Weird, the numbers are sometimes mixed : self._channel.num_tx_antennas is just a mask)
        return len(self._chirps) # self._channel.num_tx_antennas

    @property
    def num_virt_antennas(self) -> int:
        return self._channel.num_tx_antennas * self._channel.num_rx_antennas
       
    @property
    def profile(self, idx=0) -> ProfileCfg:
        return self._profiles[idx]
    
    @property
    def range_bins(self) -> np.array:
        """Create range_bins. Use number of adc_samples because otherwise range bins are incorrect."""
        range_max = self.get_range_max(visibility_factor=1.0) # Same as self.range_resolution * self.num_adc_samples

        # Substract range bias if known
        # TODO: this is a serpate setting in the config
        # if 'rangeBias' in config.profile.keys():
        #     range_bias = config.profile['rangeBias']
        # else:
        range_bias = 0
        # Create evenly spaced range_bins. self.range_resolution * self.num_range_bins gives incorrect results
        # TI is consulted for clarity
        range_bins = np.linspace(0, range_max, self.num_range_bins)
        range_bins = np.subtract(range_bins, range_bias)
        range_bins = np.maximum(range_bins, 0)
        return range_bins
   
    @property
    def range_max(self, ) -> float:
        """
            visibility_factor = 0.8 # TI specific, take SNR into account based on software version

            Returns the maximum range in meters of this config
        """
        return self.get_range_max()

    @property
    def bandwidth(self):
        """
        Args:
            num_adc_samples (int): The number of given ADC samples in a chirp
            dig_out_sample_rate (int): The ADC sample rate
            freq_slope_const (float): The slope of the freq increase in each chirp

        Returns:
                band_width (float): The bandwidth of the radar chirp config (Hz)    
        """
        freq_slope_const = self._profiles[0]._freq_slope_const # MHz/us
        dig_out_sample_rate = self._profiles[0].dig_out_sample_rate # Ms/s
        num_adc_samples = self.num_adc_samples
        freq_slope_m_hz_per_usec = freq_slope_const
        adc_sample_period_usec = 1000.0 / dig_out_sample_rate * num_adc_samples
        band_width = freq_slope_m_hz_per_usec * adc_sample_period_usec * 1e3
   
        return band_width
    
    @property
    def range_resolution(self):
        """ Calculate the range resolution for the given radar configuration

        Args:
            band_width (float): The bandwidth of the radar chirp config (Hz)    

        Returns:
            float
                range_resolution (float): The range resolution for this bin (m)
        """
        light_speed_meter_per_sec = 299792458
        range_resolution = light_speed_meter_per_sec / (2.0 * self.bandwidth)

        return range_resolution
    
    def set_channel(self, channel: ChannelCfg):
        self._channel = channel

class ConfigFile(object):
    def __init__(self, device="IWR6843AOP", show_warning=False):
        self._device = device
        self._show_warning = show_warning
        self.frameTime = 50
        self.graphFin = 1
        self.hGraphFin = 1
        self.threeD = 1
        self.lastFramePoints = np.zeros((5,1))
        self.plotTargets = 1
        self.frameNum = 0
        self.profile = {'startFreq': 60.25, 'sensorHeight':3, 'maxRange':10, 'az_tilt':0, 'elev_tilt':0, 'enabled':0}
        self.chirpComnCfg = {'DigOutputSampRate':23, 'DigOutputBitsSel':0, 'DfeFirSel':0, 'NumOfAdcSamples':128, 'ChirpTxMimoPatSel':4, 'ChirpRampEndTime':36.1, 'ChirpRxHpfSel':1}
        self.chirpTimingCfg = {'ChirpIdleTime':8, 'ChirpAdcSkipSamples':24, 'ChirpTxStartTime':0, 'ChirpRfFreqSlope':47.95, 'ChirpRfFreqStart':60}
        self.rangeRes = 0
        self.rangeAxisVals = np.zeros(int(self.chirpComnCfg['NumOfAdcSamples']/2))
        self.sensorHeight = 1.5
        self.numFrameAvg = 10
        self.configSent = 0
        self.previousFirstZ = -1
        self.yzFlip = 0
        #timer to reset fall detected message

        self.fallResetTimerOn = 0
        self.fallThresh = -0.22

        self.trackColorMap = None
        self.vitalsPatientData = []

        # class based data storage
        self._chirps = []
        self._frames = []
        self._profiles = []
        self._guimonitor = None
        self._clutter_removal = None
        self._cfarCfgs = []
        self._cfarFovCfgs = []
        self._gtrack = GTRACK()
        self._sensor = RadarSS()

    def __str__(self):
        if self._clutter_removal:
            return str(self._sensor.num_doppler_bins) + "x" + str(self._sensor.num_range_bins) + " clutter_removal=" + str(self._clutter_removal.enabled())
        else:
            return str(self._sensor.num_doppler_bins) + "x" + str(self._sensor.num_range_bins)
        
    def describe(self):
        for p in self._profiles:
            p.describe()
        for f in self._frames:
            f.describe()
        for c in self._cfarFovCfgs:
            c.describe()
        if self._guimonitor:
            self._guimonitor.describe()
        #print("Frame periocidy", self.profile[""])
        print("Transmit antennas ", int(self._sensor.num_tx_antennas))
        print(self.profile)
        

    def parseCfg(self, fname):
        with open(fname, 'r') as cfg_file:
            self.cfg = cfg_file.readlines()
        counter = 0
        chirpCount = 0
        for line in self.cfg:
            args = line.split()
            if (len(args) > 0):
                # cfarCfg
                if (args[0] == 'cfarCfg'):
                    cfar_cfg = CfarCfg()
                    cfar_cfg._subFrameIdx = int(args[1])
                    cfar_cfg._procDirection = int(args[2])
                    cfar_cfg._mode = int(args[3])
                    cfar_cfg._noiseWin = int(args[4])
                    cfar_cfg._guardLen = int(args[5])
                    cfar_cfg._divShift = int(args[6])
                    cfar_cfg._cyclic = int(args[7])
                    cfar_cfg._thresholdScale = int(args[8])
                    cfar_cfg._peakGrouping = int(args[9])
                    self._cfarCfgs.append(cfar_cfg)
                elif (args[0] == "adcCfg"):
                    adc_cfg = AdcCfg()
                    adc_cfg._numADCBits = int(args[1]) # Number of ADC bits (0  for 12-bits, 1 for 14-bits ,and 2 for 16-bits) only 16 bits supported
                    adc_cfg._adcOutputFmt = int(args[2]) # only complex is supported
                    self._adc_cfg = adc_cfg
                elif (args[0] == "adcbufCfg"):
                    adc_buf_cfg = AdcBufCfg()
                    adc_buf_cfg.parse(args)
                    self._sensor._adc_buf_cfg = adc_buf_cfg
                elif (args[0] == 'antGeometry0'):
                    ant_geometry = AntGeometry()
                    ant_geometry.parse(args)
                    self._ant_geometry0 = ant_geometry
                elif (args[0] == 'antGeometry1'):
                    ant_geometry = AntGeometry()
                    ant_geometry.parse(args)
                    self._ant_geometry1 = ant_geometry
                elif (args[0] == 'antPhaseRot'):
                    ant_phase_rot = AntPhaseRot()
                    ant_phase_rot.parse(args)
                    self._ant_phase_rot = ant_phase_rot
                elif (args[0] == 'bpmCfg'):
                    bpm_cfg = BpmCfg()
                    bpm_cfg.parse(args)
                    self._bpm_cfg = bpm_cfg
                elif (args[0] == 'calibdata'):
                    logger.error("calibdata")
                elif (args[0] == 'calibDcRangeSig'):
                    calib = CalibDcRangeSig()
                    calib.parse(args)
                    self._calib_dc_range_sig = calib
                elif (args[0] == 'clutterRemoval'):
                    cr = ClutterRemoval()
                    cr.parse(args)
                    self._clutter_removal = cr
                elif (args[0] == 'dynamic2DAngleCfg'):
                    dynamic2DAngleCfg = Dynamic2DAngleCfg()
                    dynamic2DAngleCfg.parse(args)
                    self._dynamic2DAngleCfg = dynamic2DAngleCfg
                elif (args[0] == "dynamicRACfarCfg"):
                    dynamic_racfar_cfg = DynamicRACfarCfg()
                    dynamic_racfar_cfg.parse(args)
                    self._dynamic_racfar_cfg = dynamic_racfar_cfg
                elif (args[0] == 'fineMotionCfg'):
                    fmc = FineMotionCfg()
                    fmc.parse(args)
                    self._fineMotionCfg = fmc
                elif (args[0] == "staticRACfarCfg"):
                    static_racfar_cfg = StaticRACfarCfg()
                    static_racfar_cfg.parse(args)
                    self._static_racfar_cfg = static_racfar_cfg
                elif (args[0] == "multiObjBeamForming"):
                    moobf = MultiObjBeamForming()
                    moobf.parse(args)
                    self._multiObjBeamForming = moobf
                # trackingCfg
                elif (args[0] == 'trackingCfg'):
                    if (len(args) < 5):
                        print ("Error: trackingCfg had fewer arguments than expected")
                        continue
                    logger.error("trackingConfig")
                    self.profile['maxTracks'] = int(args[4])
                    #self.trackColorMap = get_trackColors(self.profile['maxTracks'])
                    for m in range(self.profile['maxTracks']):
                        # Add track gui object
#                        mesh = gl.GLLinePlotItem()
#                        mesh.setVisible(False)
#                        self.pcplot.addItem(mesh)
#                        self.ellipsoids.append(mesh)
#                         #add track coordinate string
#                        text = GLTextItem()
#                        text.setGLViewWidget(self.pcplot)
#                        text.setVisible(False)
#                        self.pcplot.addItem(text)
#                        self.coordStr.append(text)
                        pass
                    # If we only support 1 patient, hide the other patient window
                    if (self.profile['maxTracks'] == 1):
                        #print("self.vitals[1]['pane'].setVisible(False)")
                        pass
                    # Initialize Vitals output dictionaries for each potential patient
                    for i in range (min(self.profile['maxTracks'], MAX_VITALS_PATIENTS)):
                        # Initialize 
                        patientDict = {}
                        patientDict ['id'] = i
                        patientDict ['rangeBin'] = 0
                        patientDict ['breathDeviation'] = 0
                        patientDict ['heartRate'] = []
                        patientDict ['breathRate'] = 0
                        patientDict ['heartWaveform'] = []
                        patientDict ['breathWaveform'] = []
                        self.vitalsPatientData.append(patientDict)
                elif (args[0] == 'dynamicRangeAngleCfg'):
                    drac = DynamicRangeAngleCfg()
                    drac.parse(args)
                    self._dynamicRangeAngleCfg = drac
                elif (args[0] == 'allocationParam'):
                    logger.error("allocationParam")
                elif (args[0] == 'gatingParam'):
                    logger.error("gatingParam")
                elif (args[0] == 'staticRangeAngleCfg'):
                    staticRangeAngleCfg = StaticRangeAngleCfg()
                    staticRangeAngleCfg.parse(args)
                    self._staticRangeAngleCfg = staticRangeAngleCfg
                elif (args[0] == 'compRangeBiasAndRxChanPhase'):
                    c = CompRangeBiasAndRxChanPhase()
                    c.parse(args)
                    self._comp = c
                elif (args[0] == 'SceneryParam' or args[0] == 'boundaryBox'):
                    if (len(args) < 7):
                        print ("Error: SceneryParam/boundaryBox had fewer arguments than expected")
                        continue
                    self.boundaryLine = counter
                    leftX = float(args[1])
                    rightX = float(args[2])
                    nearY = float(args[3])
                    farY = float(args[4])
                    bottomZ = float(args[5])
                    topZ = float(args[6])
                    self.profile['boundaryBox'] = [leftX, rightX, nearY, farY, bottomZ, topZ]
                    #self.addBoundBox('trackerBounds', leftX, rightX, nearY, farY, bottomZ, topZ)
                    boundary_box = BoundaryBox()
                    boundary_box._x_min = leftX
                    boundary_box._x_max = rightX
                    boundary_box._y_min = nearY
                    boundary_box._y_max = farY
                    boundary_box._z_min = bottomZ
                    boundary_box._z_max = topZ
                    self._gtrack._boundary_box = boundary_box
                elif (args[0] == 'LvdsStreamCfg'):
                    lvds = LvdsStreamCfg()
                    lvds.parse(args)
                    self._lvdsStreamCfg = lvds
                elif (args[0] == 'staticBoundaryBox'):
                    self.staticLine = counter
                    ## this is part of the tracker
                elif (args[0] == 'presenceBoundaryBox'):
                    leftX = float(args[1])
                    rightX = float(args[2])
                    nearY = float(args[3])
                    farY = float(args[4])
                    bottomZ = float(args[5])
                    topZ = float(args[6])
                
                    boundary_box = PresenceBoundaryBox()
                    boundary_box._x_min = leftX
                    boundary_box._x_max = rightX
                    boundary_box._y_min = nearY
                    boundary_box._y_max = farY
                    boundary_box._z_min = bottomZ
                    boundary_box._z_max = topZ
                    self._gtrack._presence_boundary_box = boundary_box
                elif (args[0] == 'profileCfg'):
                    if (len(args) < 12):
                        print ("Error: profileCfg had fewer arguments than expected")
                        continue
                    self.profile["id"] = int(args[1]) # 0-3
                    self.profile['startFreq'] = float(args[2])
                    self.profile['idle'] = float(args[3])
                    self.profile['adcStart'] = float(args[4])
                    self.profile['rampEnd'] = float(args[5])
                    self.profile['txOutPower'] = float(args[6])
                    self.profile['txPhaseShifer'] = float(args[7])
                    self.profile['slope'] = float(args[8])
                    self.profile['txStartTime'] = float(args[9])
                    self.profile['samples'] = float(args[10])
                    self.profile['sampleRate'] = float(args[11])
                    # TODO: we need this since there can be more profiles
                    p = ProfileCfg()
                    p.parse(args)
                    self._profiles.append(p)
                    self._sensor.add_profile(p)
                    
                elif (args[0] == 'frameCfg'):
                    f = FrameCfg()
                    if not f.parse(args):
                        continue
                    self._frames.append(f)
                    self._sensor.add_frame(f)
                    

                elif (args[0] == 'chirpCfg'):
                    # note that this is for each antenna
                    chirpCount += 1
                    c = ChirpCfg()
                    c.parse(args)
                    self._sensor.add_chirp(c)
                elif (args[0] == 'analogMonitor'):
                    # Controls the enable/disable of the various monitoring features 
                    # supported in the demos
                    self.profile["rxSaturation"] = int(args[1])
                    self.profile["sigImgBand"] = int(args[2])
                elif (args[0] == 'lowPower'):
                    # args[1] is don't care (0)
                    self.profile["lowPower"] = int(args[2])
                elif (args[0] == 'extendedMaxVelocity'):
                    # mandatory
                     self.profile['extendedMaxVelocity'] = int(args[2]) # 1 or 0, enabled or disabled
                elif (args[0] == 'sensorPosition'):
                    # sensorPosition for x843 family has 3 args
                    if(1):
                        if (len(args) < 4):
                            print ("Error: sensorPosition had fewer arguments than expected")
                            continue
                        self.profile['sensorHeight'] = float(args[1])
                        self.profile['az_tilt'] = float(args[2])
                        self.profile['elev_tilt'] = float(args[3])

                    # sensorPosition for x432 family has 5 args
                    if (0):
                        if (len(args) < 6):
                            print ("Error: sensorPosition had fewer arguments than expected")
                            continue
                        #xOffset and yOffset are not implemented in the python code yet.
                        self.profile['xOffset'] = float(args[1])
                        self.profile['yOffset'] = float(args[2])
                        self.profile['sensorHeight'] = float(args[3])
                        self.profile['az_tilt'] = float(args[4])
                        self.profile['elev_tilt'] = float(args[5])
                # Only used for Small Obstacle Detection
                elif (args[0] == 'occStateMach'):
                    numZones = int(args[1])
                    if (numZones > 2):
                        print('ERROR: More zones specified by cfg than are supported in this GUI')
                elif (args[0] == 'dfeDataOutputMode'):
                    # this is a chirp config, only mode 1 and 3 are supported
                    self.profile['dfeDataOutputMode'] = int(args[1])
                    self._sensor._dfeDataOutputMode = int(args[1])
                # Only used for Small Obstacle Detection
                elif (args[0] == 'zoneDef'):
                    if (len(args) < 8):
                        print ("Error: zoneDef had fewer arguments than expected")
                        continue
                    zoneIdx = int(args[1])
                    minX = float(args[2])
                    maxX = float(args[3])
                    minY = float(args[4])
                    maxY = float(args[5])
                    # Offset by 3 so it is in center of screen
                    minZ = float(args[6]) + self.profile['sensorHeight']
                    maxZ = float(args[7]) + self.profile['sensorHeight']

                    name = 'occZone' + str(zoneIdx)

                    self.addBoundBox(name, minX, maxX, minY, maxY, minZ, maxZ)
                elif (args[0] == 'mpdBoundaryBox'):
                    if (len(args) < 8):
                        print ("Error: mpdBoundaryBox had fewer arguments than expected")
                        continue
                    zoneIdx = int(args[1])
                    minX = float(args[2])
                    maxX = float(args[3])
                    minY = float(args[4])
                    maxY = float(args[5])
                    minZ = float(args[6])
                    maxZ = float(args[7])
                    name = 'mpdBox' + str(zoneIdx)
                    if(OFF_CHIP_PRESENCE_DETECTION_ENABLED == 1):
                        minorMotionStateMachines.append(minorBoundaryBoxStateMachineType())
                    self.addBoundBox(name, minX, maxX, minY, maxY, minZ, maxZ)

                elif (args[0] == 'chirpComnCfg'):
                    if (len(args) < 8):
                        print ("Error: chirpComnCfg had fewer arguments than expected")
                        continue
                    try:
                        self.chirpComnCfg['DigOutputSampRate'] = int(args[1])
                        self.chirpComnCfg['DigOutputBitsSel'] = int(args[2])
                        self.chirpComnCfg['DfeFirSel'] = int(args[3])
                        self.chirpComnCfg['NumOfAdcSamples'] = int(args[4])
                        self.chirpComnCfg['ChirpTxMimoPatSel'] = int(args[5])
                        self.chirpComnCfg['ChirpRampEndTime'] = 10 * float(args[6])
                        self.chirpComnCfg['ChirpRxHpfSel'] = int(args[7])
                    except Exception as e:
                        print (e)

                elif (args[0] == 'chirpTimingCfg'):
                    if (len(args) < 6):
                        print ("Error: chirpTimingCfg had fewer arguments than expected")
                        continue
                    self.chirpTimingCfg['ChirpIdleTime'] = 10.0 * float(args[1])
                    self.chirpTimingCfg['ChirpAdcSkipSamples'] = int(args[2]) << 10
                    self.chirpTimingCfg['ChirpTxStartTime'] = 10.0 * float(args[3])
                    self.chirpTimingCfg['ChirpRfFreqSlope'] = float(args[4])
                    self.chirpTimingCfg['ChirpRfFreqStart'] = float(args[5])
                elif (args[0] == 'clusterCfg'):
                    if (len(args) < 4):
                        print ("Error: clusterCfg had fewer arguments than expected")
                        continue
                    self.profile['enabled'] = float(args[1])
                    self.profile['maxDistance'] = float(args[2])
                    self.profile['minPoints'] = float(args[3])
                
                # NOTE - Major and Minor motion are not supported at once. Only major or minor motion
                # detection is currently supported.
                elif (args[0] == 'minorStateCfg' or args[0] == 'majorStateCfg'):
                    if (len(args) < 9):
                        print ("Error: minorStateCfg had fewer arguments than expected")
                        continue
                    pointThre1 = int(args[1])
                    pointThre2 = int(args[2])
                    snrThre2 = int(args[3])
                    pointHistThre1 = int(args[4])
                    pointHistThre2 = int(args[5])
                    snrHistThre2 = int(args[6])
                    histBufferSize = int(args[7])
                    minor2emptyThre = int(args[8])

                    stateMachineIdx = 0
                    boundaryBoxIdx = 0
                    if(OFF_CHIP_PRESENCE_DETECTION_ENABLED == 1):
                        for box in self.boundaryBoxes:
                            if('mpdBox' in box['name']):
                                minorMotionStateMachines[stateMachineIdx].configure(pointThre1, pointThre2, snrThre2, \
                                pointHistThre1, pointHistThre2, snrHistThre2,  histBufferSize, minor2emptyThre,\
                                float(box['boundList'][0].text()),float(box['boundList'][1].text()),\
                                float(box['boundList'][2].text()),float(box['boundList'][3].text()),\
                                float(box['boundList'][4].text()),float(box['boundList'][5].text()), boundaryBoxIdx)
                                stateMachineIdx=stateMachineIdx+1
                            boundaryBoxIdx = boundaryBoxIdx + 1
                elif (args[0] == 'cfarFovCfg'):
                    # Command for datapath to filter out detected points outside the 
                    # specified limits in the range direction or doppler direction
                    cfar_fov_cfg = CfarFovCfg()
                    cfar_fov_cfg._subFrameIdx = int(args[1])
                    cfar_fov_cfg._procDirection = int(args[2])
                    cfar_fov_cfg._min = float(args[3])
                    cfar_fov_cfg._max = float(args[4]) 
                    self._cfarFovCfgs.append(cfar_fov_cfg)
                elif (args[0] == 'fovCfg'):
                    fov_cfg = FovCfg()
                    fov_cfg.parse(args)
                    self._fov_cfg = fov_cfg
                elif (args[0] == 'maxAcceleration'):
                    ma = MaxAcceleration()
                    ma.parse(args)
                    self._gtrack._max_acceleration = ma
                # This is specifically guiMonitor for 60Lo, this parsing will break the gui when an SDK 3 config is sent
                elif (args[0] == 'guiMonitor'):
                    # Plot config message to datapath
                    self._guimonitor = GuiMonitorCfg()
                    self._guimonitor['detectedObjects'] = int(args[2])
                    self._guimonitor['logMagRange'] = int(args[3])
                    self._guimonitor['noiseProfile'] = int(args[4])
                    self._guimonitor['rangeAzimuthElevationHeatMap'] = int(args[5])
                    self._guimonitor['rangeDopplerHeatMap'] = int(args[6])
                    self._guimonitor['statsInfo'] = int(args[7])
                elif (args[0] == 'sensorStart'):
                    pass
                elif (args[0] == 'sensorStop'):
                    pass
                elif (args[0] == 'stateParam'):
                    # tracker
                    state_param = StateParam()
                    state_param.parse(args)
                    self._gtrack._state_param = state_param
                elif (args[0] == 'flushCfg'):
                    pass
                elif (args[0] == 'channelCfg'):
                    c = ChannelCfg()
                    c.parse(args)
                    self._sensor.set_channel(c)
                else:
                    if args[0][0] != '%':
                        if self._show_warning:
                            logger.warning("UNKNOWN ATH {}", args)

                        pass
                
            counter += 1

        # self.rangeRes = (3e8*(100/self.chirpComnCfg['DigOutputSampRate']))/(2*self.chirpTimingCfg['ChirpRfFreqSlope']*self.chirpComnCfg['NumOfAdcSamples'])
        self.rangeRes = (3e8*(100/self.chirpComnCfg['DigOutputSampRate'])*1e6)/(2*self.chirpTimingCfg['ChirpRfFreqSlope']*1e12*self.chirpComnCfg['NumOfAdcSamples'])
        # print("self.rangePlot.setXRange(0,(self.chirpComnCfg['NumOfAdcSamples']/2)*self.rangeRes,padding=0.01)")
     
        self.rangeAxisVals = np.arange(0, self.chirpComnCfg['NumOfAdcSamples']/2*self.rangeRes, self.rangeRes)
        self.profile["rangeIdxToMeters"] = (299792458 * self.profile['sampleRate'] * 1e3) / (2 * self.profile['slope'] * 1e12 * self._profiles[0].num_range_bins)
        self.profile["maxVelocity"] = 299792458 / (4 * self.profile['startFreq'] * 1e9 * (self.profile['idle'] + self.profile['rampEnd']) * 1e-6 * self._sensor.num_tx_antennas)

    def sensor(self) -> RadarSS:
        return self._sensor