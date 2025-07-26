"""
MiCS-5525 CO Sensor temperature dependency based on:
"MiCS Application Note 3 (Using MiCS-5525 for CO Detection)"

This script approximates the temperature/resistance curve from the plot in the PDF,
by fitting an exponential curve through it.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given points
points = [(10, 420), (20, 300), (35, 200),  (60, 100)]

x = [p[0] for p in points]
y = [p[1] for p in points]

def exp(x, a, b, c):
    return a * np.exp(-b * x) + c

# Fit the exponential function to the data points
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # there are some overflow warnings during optimization; ignore
    popt, pcov = curve_fit(exp, x, y)

def temperature_resistance_curve_exp_pointfitting(T):
    """
    temperature/resistance curve
    :param x: temperature in degrees C
    :return: R_0 in kOhm
    """
    return exp(T, *popt)

def temperature_resistance_curve_exp(T, R_25, T_25):
    """
    temperature/resistance curve, modelled as a decaying exponential curve
    :param T: temperature in degrees C
    :param R_25: fixpoint resistance at known temperature, in kOhm
    :param T_25: said known temperature, in degree C (can be 25°C, or any other)
    :return: R_0 in kOhm
    """
    b = 29.627
    a = R_25 / np.exp(-T_25 / b)

    return a*np.exp(-T/b)

class TemperatureResistanceCurveNTC:
    B = 3073  # K

    @classmethod
    def get_resistance_at_given_temperature(cls, temperature_measured, equivalent_resistance_at_infinite_temperature):
        """
        temperature/resistance curve, modelled as NTC (negative temperature coefficient) behaviour
        :param equivalent_resistance_at_infinite_temperature: parameter R_inf
        :param temperature_measured: temperature in °C
        :return: R_0 in kOhm
        """
        return equivalent_resistance_at_infinite_temperature * np.exp(cls.B / (temperature_measured + 273.15))

    @classmethod
    def get_equivalent_resistance_at_infinite_temperature(cls, resistance_measured, temperature_measured):
        """
        R_inf, the equivalent resistance at infinite temperature, is a parameter in the R/T curve.
        This function computes it from a measured pair of resistance and temperature.

        :param resistance_measured: fixpoint resistance at known temperature, in kOhm (e.g. 270 kOhm)
        :param temperature_measured: said known temperature, in degrees C (e.g. 25°C)
        :return: R_inf, in the same unit at resistance_measured
        """
        # equivalent to:
        # R_inf = R_25 * np.exp(-B / (T_25+273.15))  # kOhm
        return resistance_measured * np.exp(-cls.B / (temperature_measured+273.15))



if __name__ == '__main__':
    # plot curve
    x_range = np.linspace(0, 80, 100)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range, temperature_resistance_curve_exp_pointfitting(x_range), label='exp_opt')

    R_inf = TemperatureResistanceCurveNTC.get_equivalent_resistance_at_infinite_temperature(270, 25)
    plt.plot(x_range, TemperatureResistanceCurveNTC.get_resistance_at_given_temperature(x_range, R_inf), label='NTC')

    plt.plot(x_range, temperature_resistance_curve_exp(x_range, 270, 25), label='exp')

    plt.title('Sensor Resistance vs Temperature')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Sensor Resistance (kOhm)')
    plt.ylim(0, 500)
    plt.xlim(0, 80)
    plt.grid(True)
    plt.legend()

    function_equation = f"y_opt = {popt[0]:.2f} * exp(-{popt[1]:.2f} * x) + {popt[2]:.2f}"
    plt.text(40, 300, function_equation, fontsize=12, color='black')
    plt.show()
