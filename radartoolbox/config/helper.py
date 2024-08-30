
import tabulate
from valtes_radartoolbox.config.ti_config import ConfigFile

GLOBAL_LIGHTSPEED = 299792458
from typing import Dict, List, Union

Basic_Globals: Dict[str, Union[Dict[str, float], Dict[str, Dict[str, Union[int, float, str, Dict[str, Dict[str, Union[float, int]]], List[Union[int, float, str]]]]]]] = {
    'Constants': {
        'lightSpeed': 299792458,              # speed of light m/us   F46
        'kB': 1.38064852e-23,               # Bolzmann constant J/K, kgm^2/s^2K  F47
        'cube_4pi': (4 * 3.141592653589793) ** 3  # F48
    },
    'Platforms': {
        'xWR6843': {
            'evms': {
                'xWR6843AOP': {
                    'tx_gain': 5.2,
                    'rx_gain': 5.2,
                    'tx_power': 12
                },
                'xWR6843ISK': {
                    'tx_gain': 7,
                    'rx_gain': 7,
                    'tx_power': 12
                },
                'xWR6843ODS': {
                    'tx_gain': 5,
                    'rx_gain': 5,
                    'tx_power': 12
                },
                'xWR6843 Horn Antenna': {
                    'tx_gain': 12.8,
                    'rx_gain': 12.8,
                    'tx_power': 12
                }
            },
            'num_tx': 3,
            'num_rx': 4,
            'freq_range': [[60, 64, '60-64']],
            'adcModes': ['Complex 1x', 'Real'],
            'maximum_radar_cube_size': 768,
            'maximum_IF_bandwidth': 10,
            'maximum_sampling_frequency': 12.5,
        },
        'xWR1843': {
            'evms': {
                'xWR1843AOP': {
                    'tx_gain': 8,
                    'rx_gain': 8,
                    'tx_power': 16
                },
                'xWR1843BOOST': {
                    'tx_gain': 7,
                    'rx_gain': 7,
                    'tx_power': 12
                }
            },
            'num_tx': 3,
            'num_rx': 4,
            'freq_range': [[76, 77, '76-77'], [77, 81, '77-81']],
            'adcModes': ['Complex 1x', 'Real'],
            'maximum_radar_cube_size': 1024,
            'maximum_IF_bandwidth': 10,
            'maximum_sampling_frequency': 12.5,
        },
        'xWR1642': {
            'evms': {
                'xWR1642BOOST': {
                    'tx_gain': 10.5,
                    'rx_gain': 10.5,
                    'tx_power': 12.5
                }
            },
            'num_tx': 2,
            'num_rx': 4,
            'freq_range': [[76, 77, '76-77'], [77, 81, '77-81']],
            'adcModes': ['Complex 1x', 'Real'],
            'maximum_radar_cube_size': 768,
            'maximum_IF_bandwidth': 5,
            'maximum_sampling_frequency': 6.25,
        },
        'xWRL6432': {
            'evms': {
                'xWRL6432BOOST': {
                    'tx_gain': 5.5,
                    'rx_gain': 5.5,
                    'tx_power': 11
                }
            },
            'num_tx': 2,
            'num_rx': 3,
            'freq_range': [[57, 64, '57-64'], [60, 64, '60-64']],
            'adcModes': ['Real'],
            'maximum_radar_cube_size': 768,
            'maximum_IF_bandwidth': 5,
            'maximum_sampling_frequency': 6.25,
        },
        'xWRL1432': {
            'evms': {
                'xWRL1432BOOST': {
                    'tx_gain': 5.5,
                    'rx_gain': 5.5,
                    'tx_power': 11
                }
            },
            'num_tx': 2,
            'num_rx': 3,
            'freq_range': [[76, 77, '76-77'], [77, 81, '77-81']],
            'adcModes': ['Real'],
            'maximum_radar_cube_size': 768,
            'maximum_IF_bandwidth': 5,
            'maximum_sampling_frequency': 6.25,
        },
        'xWR1443': {
            'evms': {
                'xWR1443BOOST': {
                    'tx_gain': 9,
                    'rx_gain': 9,
                    'tx_power': 12
                }
            },
            'num_tx': 3,
            'num_rx': 4,
            'freq_range': [[76, 77, '76-77'], [77, 81, '77-81']],
            'adcModes': ['Complex 1x', 'Real'],
            'maximum_radar_cube_size': 384,
            'maximum_IF_bandwidth': 15,
            'maximum_sampling_frequency': 18.75,
        },
        'AWR1243': {
            'evms': {
                'AWR1243BOOST': {
                    'tx_gain': 7,
                    'rx_gain': 7,
                    'tx_power': 12
                }
            },
            'num_tx': 3,
            'num_rx': 4,
            'freq_range': [[76, 77, '76-77'], [77, 81, '77-81'], [76, 81, '76-81']],
            'adcModes': ['Complex 1x', 'Real'],
            'maximum_radar_cube_size': 1000000,
            'maximum_IF_bandwidth': 15,
            'maximum_sampling_frequency': 18.75,
        },
        'AWR2243': {
            'evms': {
                'AWR2243BOOST': {
                    'tx_gain': 7,
                    'rx_gain': 7,
                    'tx_power': 12
                }
            },
            'num_tx': 3,
            'num_rx': 4,
            'freq_range': [[76, 77, '76-77'], [77, 81, '77-81']],
            'adcModes': ['Complex 1x', 'Real'],
            'maximum_radar_cube_size': 1000000,
            'maximum_IF_bandwidth': 20,
            'maximum_sampling_frequency': 22.5,
        },
    }
}

Adv_Globals: Dict[str, Union[Dict[str, float], Dict[str, Dict[str, Union[List[str], List[str], int, float]]]]] = {
    'Constants': {
        'lightSpeed_mps': 299792458,    # speed of light m/us   F46
        'kB': 1.38064852e-23,           # Bolzmann constant J/K, kgm^2/s^2K  F47
        'cube_4pi': (4 * 3.141592653589793) ** 3  # F48
    },
    'Platforms': {
        'xWR6843': {
            'evms': ['xWR6843AOP', 'xWR6843ISK', 'xWR6843ODS', 'xWR6843 Horn Antenna'],
            'adcModes': ['Complex 1x', 'Real'],
            'sdkVer': 3,
            'maxInterFreq_MHz': 10
        },
        'xWRL6432': {
            'evms': ['xWRL6432BOOST'],
            'adcModes': ['Real'],
            'sdkVer': 5,
            'maxInterFreq_MHz': 5
        },
        'xWRL1432': {
            'evms': ['xWRL1432BOOST'],
            'adcModes': ['Real'],
            'sdkVer': 5,
            'maxInterFreq_MHz': 5
        },
        'xWR1843': {
            'evms': ['xWR1843AOP', 'xWR1843BOOST'],
            'adcModes': ['Complex 1x', 'Real'],
            'sdkVer': 3,
            'maxInterFreq_MHz': 10
        },
        'xWR1642': {
            'evms': ['xWR1642BOOST'],
            'adcModes': ['Complex 1x', 'Real'],
            'sdkVer': 3,
            'maxInterFreq_MHz': 5
        },
        'xWR1443': {
            'evms': ['xWR1443BOOST'],
            'adcModes': ['Complex 1x', 'Real'],
            'sdkVer': 3,
            'maxInterFreq_MHz': 15
        }
    },
    'GainsByEVM': {
        'xWR6843AOP': 5.2,
        'xWR6843ISK': 7,
        'xWR6843ODS': 5,
        'xWR6843 Horn Antenna': 13.1,
        'xWRL6432BOOST': 4,
        'xWRL1432BOOST': 4,
        'xWR1843AOP': 5.2,
        'xWR1843BOOST': 7,
        'xWR1642BOOST': 10.5,
        'xWR1443BOOST': 9
    }
}

def get_idle_time(valid_sweep_bandwidth: float) -> int:
    if valid_sweep_bandwidth <= 1000:
        return 2  # us
    elif valid_sweep_bandwidth <= 2000:
        return 5  # us
    elif valid_sweep_bandwidth <= 3000:
        return 6  # us
    else:
        return 7  # us

def get_maximum_beat_frequency(ramp_slope: float, maximum_detectable_range: float, light_speed: float) -> float:
    return 2000000 * ramp_slope * maximum_detectable_range / light_speed

def get_maximum_radar_cube_size(device: str) -> int:
    """
    Get the device's max radar cube storage size in kB, which is equivalent to 'L3 RAM size' in the data sheet.
    
    Args:
        device (str): The radar device for which the maximum radar cube size is needed.
        
    Returns:
        int: The maximum radar cube size in kilobytes.
    """
    return Basic_Globals['Platforms'][device]['maximum_radar_cube_size']

def get_ramp_slope_init(valid_sweep_bandwidth, chirp_time_temp):
    return valid_sweep_bandwidth / chirp_time_temp

def get_ramp_slope_parameter(ramp_slope_init):
    return round((ramp_slope_init * (1 << 26) * 1000000) / (3600000000 * 900))

def get_typical_detected_object_to_rcs(typical_detected_object: str) -> float:
    if typical_detected_object == 'Truck':
        return 100.0
    elif typical_detected_object == 'Car':
        return 5.0
    elif typical_detected_object == 'Motorcycle':
        return 3.2
    elif typical_detected_object == 'Adult':
        return 1.0
    elif typical_detected_object == 'Child':
        return 0.5
    else:
        return None

def get_valid_sweep_bandwidth(light_speed, range_resolution):
    # input is range resolution in cm
    return light_speed / (2 * range_resolution * 0.01 * 1000000)

class ChirpConfigurationParameters(object):
    """
        This is the describe similar to the https://dev.ti.com/gallery/view/mmwave/mmWaveSensingEstimator/ver/2.3.0/app/
    """
    def __init__(self):
        pass

    def describe(self, range_resolution):
        #Starting frequency of the chirp ramp.
      ###  print(self._config._sensor.profile._start_freq)
        #Slope of the chirp as it ramps frequency.  Frequency Slope must be equal to or less than 100 MHz/us.
       ### print(self._config._sensor.profile._freq_slope_const)
        
        #Register-level parameter to represent the Frequency Slope.
        valid_sweep_bandwidth = get_valid_sweep_bandwidth(GLOBAL_LIGHTSPEED, range_resolution);
        
        print("valid_sweep_bandwidth", valid_sweep_bandwidth)
        #rsi = get_ramp_slope_init(valid_sweep_bandwidth, chirp_time_temp)
       # print(get_ramp_slope_parameter(rsi))
        print("max_radar_cube", get_maximum_radar_cube_size("xWR6843"), "kB")