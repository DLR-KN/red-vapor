import os
import pickle
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.errors import PerformanceWarning
from scipy.signal import find_peaks, peak_widths

from mics.humidity_resistance_curve import HumidityResistanceCurve
from mics.concentration_resistance_curve import ConcentrationResistanceCurve
from mics.temperature_resistance_curve import TemperatureResistanceCurveNTC
from util import helpers
from util.constants import EXPERIMENT_CLASSES, MICS_SENSOR_TYPES, PID_SENSOR_TYPES, ALL_SENSOR_TYPES, SENSOR_SPECIMEN, \
    DATASET_OUTPUT_PATH, LAYER_SENSOR_SPECIMEN, UPSTREAM_SENSOR_SPECIMEN


def detect_steps(data, upstream_idx):
    """
    Some of the sensor signals show steps at distinct points in time.
    We attribute this to the fact that the upstream sensors did not have an input voltage that is as clean as for
    the main sensor platforms; i.e. the ADC reference voltage may have jumped. But since these jumps are straightforward
    to detect, we can compensate them by re-attaching the signal to be smooth at the point of the jump.

    :param upstream_idx: upstream index
    :param data: timeseries
    :return: list of jumps
    """

    # convolve timeseries with a step response
    step_length = 100
    step = np.hstack((-np.ones(step_length), np.ones(step_length)))
    conv_step = np.convolve(data, step, mode='valid')

    # locate very significant spikes
    negative_spikes = find_peaks(conv_step, height=7 * conv_step.std(), distance=100, wlen=200, width=(None, 7), rel_height=0.01)
    positive_spikes = find_peaks(-conv_step, height=7 * conv_step.std(), distance=100, wlen=200, width=(None, 7), rel_height=0.01)
    all_spike_indices = np.concat((negative_spikes[0], positive_spikes[0]))
    all_spike_heights = np.concat((-negative_spikes[1]['peak_heights'], positive_spikes[1]['peak_heights']))

    # spikes in convolution let us derive times and amplitudes of steps
    steps = []
    for peak_index, peak_height in zip(all_spike_indices, all_spike_heights):
        step_time = data.index[peak_index + step_length]  # TODO: plus 1?
        step_amplitude = peak_height / (len(step) / 2)
        # print(f"detected step: {data.name}\t{step_time}\t{step_amplitude}")
        steps.append({
            'step_time': step_time,
            'step_amplitude': step_amplitude,
            'upstream_idx': upstream_idx,
        })

        """ visualize the classification criterion for 'step/no step' """
        # peak_width = peak_widths(-np.sign(step_amplitude)*conv_step, [peak_index], wlen=200, rel_height=0.01)[0]
        # plt.plot(conv_step[peak_index - 100: peak_index + 100])
        # plt.title(f"peak width: {peak_width[0]}")
        # plt.show()

    return steps


def voltage_to_resistance(data, sensor_type):
    """
    Convert voltages to resistances R_S, depending on whether the load resistor is on the high or low side
    :param data:
    :param sensor_type:
    :return:
    """
    V_in = 5  # volt

    match sensor_type:
        case 'MiCS5524':
            R_load = 10_000  # ohm
            # R_load is 'low-side'
            return ((V_in / data) - 1) * R_load
        case 'MiCS6814_NO2' | 'MiCS6814_NH3':
            R_load = 47_000  # ohm
            # R_load is 'high-side'
            return (data * R_load) / (V_in - data)
        case _:
            raise ValueError("Invalid sensor type")


if __name__ == '__main__':
    """
    Read and preprocess sensor data from wind tunnel experiments.
    This preprocessing routine is valid for the PID, MiCS5524 and MiCS6814 sensors
    """

    """ load data """
    experiments = helpers.load_experiments()  # dict(Experiment Types) of list(Experiments) of dict(Experiment)

    # display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 100000)

    """ convert to MultiIndex Dataframe """
    # TODO: consider moving this into load_experiments()
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in SENSOR_SPECIMEN:
                df = experiment[key][idx]
                for sensor_type in ALL_SENSOR_TYPES:
                    df.rename(columns={sensor_type: (sensor_type, 'voltage')}, inplace=True)
                df.columns = pd.MultiIndex.from_tuples(
                    [col if isinstance(col, tuple) else (col, '') for col in df.columns])

    """ anomaly correction """
    # some signals show artefacts/anomalies. correct these first
    print("Detecting Steps in Upstream PID Voltage Signal:")
    upstream_voltage_steps = []
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in UPSTREAM_SENSOR_SPECIMEN:
                df = experiment[key][idx]
                for sensor_type in PID_SENSOR_TYPES:
                    steps = detect_steps(df[(sensor_type, 'voltage')], idx)
                    upstream_voltage_steps += steps

                    # correct for steps in upstream sensor PID voltage
                    voltage_sanitized = df[(sensor_type, 'voltage')].copy()
                    for step in steps:
                        # save 'before' series
                        window = pd.Timedelta("10min")
                        before = voltage_sanitized[step["step_time"] - window: step["step_time"] + window].copy()

                        # correct
                        voltage_sanitized.loc[step["step_time"]:] -= step["step_amplitude"]

                        # show plot, comparing 'before' and 'after'
                        after = voltage_sanitized[step["step_time"] - window: step["step_time"] + window].copy()
                        plt.plot(after, label="after")
                        plt.plot(before, label="before")
                        plt.plot(before.rolling('10s').mean())
                        plt.plot(after.rolling('10s').mean())
                        plt.legend()
                        plt.title(f"[Anomaly Correction]\nUpstream Sensor {idx} PID Voltage Step at {step["step_time"]}")
                        plt.show()

                    df[(sensor_type, 'voltage_sanitized')] = voltage_sanitized

    upstream_voltage_steps = pd.DataFrame(upstream_voltage_steps)
    print(upstream_voltage_steps.assign(step_amplitude=upstream_voltage_steps['step_amplitude'] * 1e3))
    print(f"Total: {len(upstream_voltage_steps)} steps in upstream sensor PID voltage (unit: mV)")
    print()

    """ MiCS: voltage to resistance """
    # ADCs report voltages, measured quantity are sensor element resistances
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in SENSOR_SPECIMEN:
                df = experiment[key][idx]
                for sensor_type in MICS_SENSOR_TYPES:
                        df[(sensor_type, 'resistance')] = voltage_to_resistance(df[(sensor_type, 'voltage')], sensor_type)

    """ MiCS: temperature and humidity compensation """
    # build column index
    col_subheaders = ["t_max", "R0", "T0", "RH0", "R0_25C_50RH", "R0_inf_at50"]
    column_tuples = []
    def mics_col_name(i):
        return f"E{i + 1}"
    for exp_class in EXPERIMENT_CLASSES:
        exp_col_names = [f"E{i + 1}" for i in range(len(experiments[exp_class]))]
        for col in exp_col_names:
            for sub in col_subheaders:
                column_tuples.append((exp_class, col, sub))
    columns = pd.MultiIndex.from_tuples(column_tuples, names=['Experiment Class', 'Experiment', ''])
    # build row index
    rows = pd.MultiIndex.from_product([MICS_SENSOR_TYPES, SENSOR_SPECIMEN], names=['Sensor Type', 'Specimen'])
    # create dataframe
    mics_calibration = pd.DataFrame(index=rows, columns=columns)

    mean_window = '2s'  # filter parameter

    # for each MOS sensor specimen:
    # 1. find time of maximum resistance, t_max. take this as "clean-air" reference.
    #    for noise suppression, we sample a small window around this t_max.
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in SENSOR_SPECIMEN:
                df = experiment[key][idx]
                for sensor_type in MICS_SENSOR_TYPES:
                    t_max = df[(sensor_type, 'resistance')].rolling(mean_window).mean().idxmax()
                    # TODO: should we restrict this to the first ~15min? there are some weird t_max
                    # TODO: some NO2 seem way too low
                    mics_calibration.loc[(sensor_type, (key, idx)), (exp_class, mics_col_name(exp_idx), 't_max')] = t_max
                    mics_calibration.loc[(sensor_type, (key, idx)), (exp_class, mics_col_name(exp_idx), 'R0')] = df[(sensor_type, 'resistance')].rolling(mean_window).mean()[t_max]
                    mics_calibration.loc[(sensor_type, (key, idx)), (exp_class, mics_col_name(exp_idx), 'T0')] = df['bme680-temp'].rolling(mean_window).mean()[t_max]
                    mics_calibration.loc[(sensor_type, (key, idx)), (exp_class, mics_col_name(exp_idx), 'RH0')] = df['bme680-humidity'].rolling(mean_window).mean()[t_max]
    print("MiCS: extract time of maximum resistance t_max, along with R, T, RH at t_max:")
    print(mics_calibration)
    # TODO: look at tunnel flushing periods in between experiments as a source for R_0


    # 2. take the resistance at t_max, and correct it to obtain R0_25C_50RH
    #  a. correct for humidity at t_max:
    #  b. correct for temperature at t_max:
    #   result: R0_25C_50RH = equivalent clean-air resistance of this specimen @25°C, 50%RH
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in SENSOR_SPECIMEN:
                for sensor_type in MICS_SENSOR_TYPES:
                    with warnings.catch_warnings():
                        # mics_calibration is sorted semantically, not alphabetically. ignore performance penalty warning.
                        warnings.simplefilter("ignore", PerformanceWarning)
                        entry = mics_calibration.loc[(sensor_type, (key, idx)), (exp_class, mics_col_name(exp_idx))]
                    # a. correct for humidity
                    R50_at_T0 = HumidityResistanceCurve.get_equivalent_resistance_at_humidity_50(entry['R0'].item(), entry['RH0'].item())
                    R_humid_comp = HumidityResistanceCurve.get_resistance_given_humidity(50, R50_at_T0)
                    # b. correct for temperature
                    R0_inf_at50 = TemperatureResistanceCurveNTC.get_equivalent_resistance_at_infinite_temperature(R_humid_comp, entry['T0'].item())
                    R0_25C_50RH = TemperatureResistanceCurveNTC.get_resistance_at_given_temperature(25, R0_inf_at50)
                    # save values
                    mics_calibration.loc[(sensor_type, (key, idx)), (exp_class, mics_col_name(exp_idx), 'R0_inf_at50')] = R0_inf_at50
                    mics_calibration.loc[(sensor_type, (key, idx)), (exp_class, mics_col_name(exp_idx), 'R0_25C_50RH')] = R0_25C_50RH
                    # R0_25C_50RH is not required for further computations; but it is still an interesting quantity in itself
    print("MiCS: computed equivalent R0 @ 25°C, 50%RH (and temperature curve parameter R0_inf at that point):")
    print(mics_calibration)

    # attach to individual experiments
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            experiment['mics_calibration'] = mics_calibration[exp_class][mics_col_name(exp_idx)]


    # 3. from this fixed point, create a signal R0[t] as a function of the measured T and RH
    #  a. take R0_inf_at50 & measured T -> R0_temp_comp
    #  b. take R0_temp_comp & measured RH -> R0_temp_humid_comp
    #   result: R0[t] = R0_temp_humid_comp
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in SENSOR_SPECIMEN:
                df = experiment[key][idx]
                for sensor_type in MICS_SENSOR_TYPES:
                    R0_inf = mics_calibration.loc[(sensor_type, (key, idx)), (exp_class, mics_col_name(exp_idx), 'R0_inf_at50')].item()
                    # apply smoothing to T and TH
                    Ts = df['bme680-temp'].rolling(mean_window).mean()
                    RHs = df['bme680-humidity'].rolling(mean_window).mean()
                    R0_temp_comp = TemperatureResistanceCurveNTC.get_resistance_at_given_temperature(Ts, R0_inf)
                    R0_temp_humid_comp = HumidityResistanceCurve.get_resistance_given_humidity(RHs, R0_temp_comp)

                    # store R0[T] = synthetic gas-free resistance signal, function of T and RH
                    df[(sensor_type, 'R_gas-free')] = R0_temp_humid_comp


    # 4. use R0[t] to compute R[t] / R0[t]. use this ratio to compute the ppm level
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in SENSOR_SPECIMEN:
                df = experiment[key][idx]
                for sensor_type in MICS_SENSOR_TYPES:
                    # compute Rs/R0
                    df[(sensor_type, 'Rs/R0')] = df[(sensor_type, 'resistance')] / df[(sensor_type, 'R_gas-free')]
                    # from Rs/R0, compute ppm
                    df[(sensor_type, 'ppm')] = ConcentrationResistanceCurve.get_concentration(
                        Rs=df[(sensor_type, 'resistance')],
                        R0=df[(sensor_type, 'R_gas-free')],
                        sensor_type=sensor_type)

    

    """ PID: get offset voltages and subtract them """
    # 1. determine offset per sensor
    # 2. subtract it
    # This is as described in Alphasense AAN-302-04
    # There it is also mentioned that PID sensitivity increases ever-so-slightly with temperature,
    # but that this is safe to ignore (20°C -> 50°C = +1%)

    # build column index
    column_tuples = []
    def pid_offs_col_name(i):
        return f"E{i + 1} (mV)"
    for exp_class in EXPERIMENT_CLASSES:
        exp_col_names = [pid_offs_col_name(i) for i in range(len(experiments[exp_class]))]
        for col in exp_col_names:
            column_tuples.append((exp_class, col))
    columns = pd.MultiIndex.from_tuples(column_tuples, names=['Experiment Class', 'Experiment'])
    # build row index
    rows = pd.MultiIndex.from_product([PID_SENSOR_TYPES, SENSOR_SPECIMEN], names=['Sensor Type', 'Specimen'])
    # create dataframe
    pid_offset_voltages = pd.DataFrame(index=rows, columns=columns)

    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in SENSOR_SPECIMEN:
                df = experiment[key][idx]
                for sensor_type in PID_SENSOR_TYPES:
                    if key == 'upstream':
                        voltage_signal = df[(sensor_type, 'voltage_sanitized')]
                    else:
                        voltage_signal = df[(sensor_type, 'voltage')]
                    offset = voltage_signal.rolling(f'5s').mean().min() * 1000
                    pid_offset_voltages.loc[(sensor_type, (key, idx)), (exp_class, pid_offs_col_name(exp_idx))] = offset

                    df[(sensor_type, 'voltage_offset_corrected')] = voltage_signal - offset/1000
    print("\nPID Offset Voltages:")
    print(pid_offset_voltages)

    # attach to individual experiments
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            experiment['pid_offset_voltages'] = pid_offset_voltages[exp_class][pid_offs_col_name(exp_idx)]

    """ PID: compute and apply correction factors """
    # build column index
    column_tuples = []
    def pid_factor_col_name(i):
        return f"E{i + 1}"
    for exp_class in EXPERIMENT_CLASSES:
        exp_col_names = [pid_factor_col_name(i) for i in range(len(experiments[exp_class]))]
        for col in exp_col_names:
            column_tuples.append((exp_class, col))
    columns = pd.MultiIndex.from_tuples(column_tuples, names=['Experiment Class', 'Experiment'])
    # build row index
    rows = pd.MultiIndex.from_product([PID_SENSOR_TYPES, SENSOR_SPECIMEN], names=['Sensor Type', 'Specimen'])
    # create dataframe
    pid_correction_factors = pd.DataFrame(index=rows, columns=columns)

    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for sensor_type in PID_SENSOR_TYPES:
                if exp_class in ["Purging_Runs", "Fly-Through_Experiments"]:
                    # PID correction factor computation does not make sense in (zero-concentration) purging runs,
                    # or in Fly-Through Experiments where we don't have a meaningful increase in concentration until
                    # the first spike
                    pid_correction_factors.loc[sensor_type, (exp_class, pid_factor_col_name(exp_idx))] = 1
                    continue

                # find first sharp rise in layer 0
                s = experiment['layers'][0][sensor_type]['voltage_offset_corrected']
                ds_dt = s.rolling('5s').mean().diff() / s.index.to_series().diff().dt.total_seconds()
                threshold = ds_dt.abs().mean() * 10
                sharp_rises = ds_dt[ds_dt > threshold]
                if sharp_rises.empty:
                    raise RuntimeError("could not find first sharp rise in layer 0 PID signal")
                first_pid_rise_time = sharp_rises.index[0]
                one_minute_before = first_pid_rise_time - pd.Timedelta(minutes=1)
                # ... found it. one_minute_before is the time we consider "still in clean air" - before the probe has entered
                # the plume for the first time

                # but the background concentration will have risen some
                # so take this increased background concentration to perform a primitive 2-point calibration,
                # in order to get comparable slopes (sensitivities [mV/ppm]) for each PID sensor specimen
                voltages = []
                for key, idx in SENSOR_SPECIMEN:
                    df = experiment[key][idx]
                    v_corr = df[sensor_type]['voltage_offset_corrected']
                    v_at_point = v_corr.rolling('5s').mean().iloc[s.index.get_indexer([one_minute_before], method='nearest')[0]]
                    voltages.append(v_at_point)
                mean_voltage = np.mean(voltages)  # average over all PIDs
                # mean_voltage = np.mean(voltages[0:2])  # average over the 2 upstream PIDs
                pid_correction_factors.loc[sensor_type, (exp_class, pid_factor_col_name(exp_idx))] = mean_voltage / voltages
    print("\nPID Correction Factors:")
    print(pid_correction_factors)

    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for sensor_type in PID_SENSOR_TYPES:
                # attach to individual experiments
                experiment['pid_correction_factors'] = pid_correction_factors.loc[sensor_type, (exp_class, pid_factor_col_name(exp_idx))]  # [columns[exp_idx]]
                # apply correction factors
                for key, idx in SENSOR_SPECIMEN:
                    df = experiment[key][idx]
                    df[(sensor_type, 'voltage_scaled')] = (df[(sensor_type, 'voltage_offset_corrected')]
                                                           * experiment['pid_correction_factors'][(key, idx)])

    """ PID: get concentrations """
    # convert voltage to ppm

    # Sensitivity (linear range) Isobutylene: > 25 mV/ppm, typical range 25 to 75 mV/ppm, average ~55mV/ppm
    # PID-AH2 sensor lamp type: 10.6 eV
    # Response Factors (RF): Ethanol = 11, Propane-1,2-diol = 3
    # Gas Mixture RF for relative proportions: CF(mix) = 1 / [(a/CF(A) + b/CF(B) + c/CF(C)…]
    # our mixture: 40ml Ethanol, 2000ml PG
    # TODO: Ethanol was 20ml for some experiments! everything until Wednesday 16:24h local time
    # TODO: can we obtain better sensitivity values for our sensors?
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in SENSOR_SPECIMEN:
                df = experiment[key][idx]
                for sensor_type in PID_SENSOR_TYPES:
                    # Formula: voltage [V] * 1000 mV/V / sensitivity [mV/ppm] * RF
                    df[(sensor_type, 'ppm_ethanol')] = df[(sensor_type, 'voltage_scaled')] * 1000 / 55 * 11
                    df[(sensor_type, 'ppm_PG')] = df[(sensor_type, 'voltage_scaled')] * 1000 / 55 * 3
                    df[(sensor_type, 'ppm')] = df[(sensor_type, 'voltage_scaled')] * 1000 / 55 * 1/(40/2040/11 + 2000/2040/3)

    """ eliminate background concentration """
    # compute mean of background sensor ppm, and subtract this from each layer
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in LAYER_SENSOR_SPECIMEN:
                df = experiment[key][idx]
                for sensor_type in ALL_SENSOR_TYPES:
                    # re-index upstreams to the index of this series
                    layer_series_index = df[(sensor_type, 'ppm')].index
                    upstream_interp = [None, None]
                    for upstream_idx in [0, 1]:
                        upstream_interp[upstream_idx] = experiment['upstream'][upstream_idx][sensor_type]['ppm']
                        upstream_interp[upstream_idx] = upstream_interp[upstream_idx].reindex(layer_series_index.union(upstream_interp[upstream_idx].index))
                        upstream_interp[upstream_idx] = upstream_interp[upstream_idx].interpolate(method='time')
                        upstream_interp[upstream_idx] = upstream_interp[upstream_idx].loc[layer_series_index]
                    # TODO: check why this index has a NaN-tail but not a NaN-head!

                    # compute mean of the two upstream sensors
                    mean_upstream = (upstream_interp[0] + upstream_interp[1]) / 2

                    # subtract a slow rolling mean of the averaged upstream sensors from the layer series
                    df[(sensor_type, 'ppm_relative')] = df[(sensor_type, 'ppm')] - mean_upstream.rolling('30s').mean()

    """ export as pickle """
    print("Example DataFrame:")
    print("Upstream 0")
    print(helpers.group_columns_by_top_level(experiments["Sampling_Experiments"][0]['upstream'][0])[100:-99])
    print("Layer 0")
    print(helpers.group_columns_by_top_level(experiments["Sampling_Experiments"][0]['layers'][0])[100:-99])

    # group all dataframe columns by top level column name, inplace
    for exp_class in EXPERIMENT_CLASSES:
        for exp_idx, experiment in enumerate(experiments[exp_class]):
            for key, idx in SENSOR_SPECIMEN:
                df = experiment[key][idx]
                experiment[key][idx] = helpers.group_columns_by_top_level(df)

    # export Python pickle
    with open(os.path.join(DATASET_OUTPUT_PATH, 'experiments.pkl'), "wb") as experiments_file:
        pickle.dump(experiments, experiments_file)
