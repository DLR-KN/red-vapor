"""
MiCS Sensor ppm/resistance curve.
Approximates the curve from the MiCS-5524 / MiCS-6814 datasheets, which show the relation ppm -> Rs/R0.
This is done using the Ethanol curve for the MiCS-5524,
the NO2 curve for the MiCS-6814 OX=NO2 channel,
and the Ethanol curve for the MiCS-6814 NH3 channel.
A straight line is fit through the log-log plot.

Note: per the datasheets, the sensors detect the substances in the following ranges:
MiCS-5524 Ethanol: 10-500ppm
MiCS-6814 ammonia (NH3): 1-500ppm
MiCS-6814 NO2: 0.05-10ppm
"""

import numpy as np

class ConcentrationResistanceCurve:
    # given points from the datasheets
    RED_Ethanol = [(5, 0.5), (60, 0.1)]  # reducing channel, response to ethanol
    OX_NO2 = [(0.08, 0.5), (0.6, 4)]  # oxidizing channel, response to NO2
    NH3_Ethanol = [(200, 0.08), (6, 0.3)]  # NH3 channel, response to ethanol


    # Function to plot a linear curve in log-log
    @classmethod
    def slope(cls, points):
        """
        Outputs the coefficients of a function y = mx + b, in arbitrary 'plot-space' units:
        x is in log(ppm), y is in log(Rs/R0)
        :param points: fixed points from the datasheet plot
        :return: coefficients b, m
        """
        x1, y1 = points[0]
        x2, y2 = points[1]
        m = (np.log(y2) - np.log(y1)) / (np.log(x2) - np.log(x1))
        b = np.log(y1) - m * np.log(x1)
        return b, m


    # Calculate the concentration for a given Rs/R0 value
    @classmethod
    def get_concentration(cls, Rs, R0, sensor_type):
        """
        Given some Rs/R0, use the fitted line to recover the ppm.
        1. y = log(Rs/R0)
        2. compute x via y = mx + b
        3. ppm = e^x

        :param Rs: Rs in the same unit as R0 (e.g. kOhm)
        :param R0: R0 in the same unit as Rs (e.g. kOhm)
        :param sensor_type:
        :return:
        """
        match sensor_type:
            case 'MiCS6814_NH3':
                b, m = cls.slope(cls.NH3_Ethanol)
            case 'MiCS6814_NO2':
                b, m = cls.slope(cls.OX_NO2)
            case 'MiCS5524':
                b, m = cls.slope(cls.RED_Ethanol)
            case _:
                raise ValueError("Invalid sensor type")

        return np.exp((np.log(Rs/R0) - b) / m)
