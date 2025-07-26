import numpy as np
from matplotlib import pyplot as plt

class HumidityResistanceCurve:
    k = -0.005  # -0.5% of resistance is lost per % of relative humidity

    @classmethod
    def get_resistance_given_humidity(cls, measured_humidity, resistance_at_humidity_50):
        """
        Take some resistance that has been measured at 50%RH, and from it derive the resistance in another
        humidity setpoint, as given by another measured relative humidity value.
        Assumes a constant slope of -0.5% of resistance lost per % of relative humidity, with the reference point
        being R at 50%RH.

        :param measured_humidity: the new RH setpoint. 0 - 100 [%RH]
        :param resistance_at_humidity_50: the resistance at 50%RH [kOhm]
        :return: the resistance in the new resistance setpoint [kOhm]
        """
        return resistance_at_humidity_50 * (1 + cls.k * (measured_humidity - 50))

    @classmethod
    def get_equivalent_resistance_at_humidity_50(cls, resistance_measured, humidity_measured):
        """
        The function get_resistance_given_humidity requires a resistance value measured at 50%RH. If this has not been
        measured on a real sensor specimen, it can be derived from another point on the curve, by providing the
        resistance measured at this point and the humidity measured at this point.

        :param resistance_measured: measured resistance [kOhm]
        :param humidity_measured: measured humidity 0 - 100 [%RH]
        :return: The equivalent resistance this sensor would read at 50%RH, according to the model.
        """
        # equivalent to: resistance_measured / ( 1 + k * (humidity_measured - 50) )  with the same k
        return resistance_measured / cls.get_resistance_given_humidity(humidity_measured, 1)

if __name__ == '__main__':
    rh_range = np.linspace(0, 100)

    resistance_at_humidity_50 = 500
    humidity_measured = 60
    resistance_measured = HumidityResistanceCurve.get_resistance_given_humidity(humidity_measured, resistance_at_humidity_50)
    R50 = HumidityResistanceCurve.get_equivalent_resistance_at_humidity_50(resistance_measured, humidity_measured)
    print(f"R50 simulated: {resistance_at_humidity_50}, R60 simulated: {resistance_measured}, R50 back-compensated: {R50}")

    plt.figure(figsize=(10, 6))
    plt.plot(rh_range, HumidityResistanceCurve.get_resistance_given_humidity(rh_range, 270), label='RH')
    plt.plot(rh_range, HumidityResistanceCurve.get_resistance_given_humidity(rh_range, 1000), label='RH')
    plt.plot(rh_range, HumidityResistanceCurve.get_resistance_given_humidity(rh_range, 500), label='RH')
    plt.plot(rh_range, HumidityResistanceCurve.get_resistance_given_humidity(rh_range, 100), label='RH')

    plt.title('Sensor Resistance vs Relative Humidity')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Sensor Resistance (kOhm)')
    plt.ylim(0, 1300)
    plt.xlim(0, 100)
    plt.grid(True)
    plt.legend()

    plt.show()

