import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from mics.humidity_resistance_curve import HumidityResistanceCurve
from mics.temperature_resistance_curve import TemperatureResistanceCurveNTC

if __name__ == '__main__':
    # let these be our clean-air measurements:
    measured_resistance = 270  # kOhm
    measured_temperature = 10  # °C
    measured_humidity = 70  # %RH

    """ forward test: from this measurement, calculate what we would get at 25°C, 50%RH """
    # temperature first, humidity second
    R_inf_at10 = TemperatureResistanceCurveNTC.get_equivalent_resistance_at_infinite_temperature(measured_resistance, measured_temperature)
    R_temp_comp = TemperatureResistanceCurveNTC.get_resistance_at_given_temperature(25, R_inf_at10)

    R50 = HumidityResistanceCurve.get_equivalent_resistance_at_humidity_50(R_temp_comp, measured_humidity)
    R_temp_humid_comp = HumidityResistanceCurve.get_resistance_given_humidity(50, R50)

    # humidity first, temperature second
    R50_at_10 = HumidityResistanceCurve.get_equivalent_resistance_at_humidity_50(measured_resistance, measured_humidity)
    R_humid_comp = HumidityResistanceCurve.get_resistance_given_humidity(50, R50_at_10)

    R_inf = TemperatureResistanceCurveNTC.get_equivalent_resistance_at_infinite_temperature(R_humid_comp, measured_temperature)
    R_humid_temp_comp = TemperatureResistanceCurveNTC.get_resistance_at_given_temperature(25, R_inf)
    # since we want to save R_inf_at50 and resistance_at_humidity_50 in the end, this is slightly preferable

    assert R_temp_humid_comp == R_humid_temp_comp, \
        "Order of operations (temperature & humidity compensation) does not matter"
    print(f"R0 at 25°C, 50% RH: {R_humid_temp_comp} kOhm")


    """ plot it on a 2D grid of T & RH """
    T = np.arange(0, 80, 1)
    RH = np.arange(0, 100, 1)
    Ts, RHs = np.meshgrid(T, RH)

    # R0 -> humidity comp -> temp comp
    R0_humid_comp = HumidityResistanceCurve.get_resistance_given_humidity(RHs, R_humid_temp_comp)
    R_inf = TemperatureResistanceCurveNTC.get_equivalent_resistance_at_infinite_temperature(R0_humid_comp[:,0], 25)
    R_inf = np.repeat(R_inf[:, np.newaxis], 80, axis=1)
    R0_humid_temp_comp = TemperatureResistanceCurveNTC.get_resistance_at_given_temperature(Ts, R_inf)

    # R0 -> temp comp -> humidity comp
    R0_inf = TemperatureResistanceCurveNTC.get_equivalent_resistance_at_infinite_temperature(R_humid_temp_comp, 25)
    R0_temp_comp = TemperatureResistanceCurveNTC.get_resistance_at_given_temperature(Ts, R0_inf)
    R0_temp_humid_comp = HumidityResistanceCurve.get_resistance_given_humidity(RHs, R0_temp_comp)
    # this computation is simpler - especially when we consider that we can have R0_inf_at50 saved (it's a constant)

    assert np.all(np.isclose(R0_temp_humid_comp, R0_humid_temp_comp)), \
        "Order of operations (temperature & humidity compensation) does not matter"

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(Ts, RHs, R0_temp_humid_comp, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Humidity')
    ax.set_zlabel('R')
    ax.view_init(elev=30, azim=20)
    plt.show()

    print()
