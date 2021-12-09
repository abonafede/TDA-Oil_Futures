"""Plot functions """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_crash_detections(
    start_date,
    end_date,
    threshold,
    buy_threshold,
    distances,
    time_index_derivs,
    price_resampled_derivs,
    metric_name
):

    # calculate rolling mean, min, max of homological derivatives
    rolled_mean_h = pd.Series(distances).rolling(20, min_periods=1).mean()
    rolled_min_h = (
        pd.Series(distances)
        .rolling(len(distances), min_periods=1)
        .min()
    )
    rolled_max_h = (
        pd.Series(distances)
        .rolling(len(distances), min_periods=1)
        .max()
    )

    # normalise the time series values to lies within [0, 1]
    probability_of_crash_h = (rolled_mean_h - rolled_min_h) / (
        rolled_max_h - rolled_min_h
    )

    # define time intervals to plots
    is_date_in_interval = (time_index_derivs > pd.Timestamp(start_date)) & (
        time_index_derivs < pd.Timestamp(end_date)
    )
    probability_of_crash_h_region = probability_of_crash_h[is_date_in_interval]
    time_index_region = time_index_derivs[is_date_in_interval]
    resampled_close_price_region = price_resampled_derivs.loc[is_date_in_interval]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(time_index_region, probability_of_crash_h_region, color="#1f77b4")
    plt.axhline(y=threshold, linewidth=2, color='#ff7f0e', linestyle='--', label='Sell Threshold')
    plt.axhline(y=buy_threshold, linewidth=2, color='red', linestyle='--', label='Buy Threshold')
    plt.title(f"Crash Indicator Based on {metric_name}")
    plt.legend(loc="best", prop={"size": 10},)

    plt.subplot(1, 2, 2)
    plt.plot(
        resampled_close_price_region[probability_of_crash_h_region.values > threshold],
        '#ff7f0e', marker='.', linestyle='None', markersize=4
    )

    plt.plot(
        resampled_close_price_region[probability_of_crash_h_region.values <= threshold],
        color="#1f77b4", marker='.', linestyle='None', markersize=4
    )
    plt.plot(
        resampled_close_price_region[probability_of_crash_h_region.values <= buy_threshold],
        color='red', marker='.', linestyle='None', markersize=4
    )

    plt.title("Close Price")
    plt.legend(
        [
            "Crash threshold > {0}".format(round(threshold, 3)),
            "Crash threshold ≤ {0}".format(round(threshold, 3)),
        ],
        loc="best",
        prop={"size": 10},
    )
    plt.show()


def plot_crash_comparisons(
    start_date,
    end_date,
    threshold,
    buy_threshold,
    distances_1,
    distances_2,
    time_index_derivs,
    price_resampled_derivs,
):

    # calculate rolling mean, min, max of homological derivatives
    rolled_mean_1 = pd.Series(distances_1).rolling(20, min_periods=1).mean()
    rolled_min_1 = (
        pd.Series(distances_1)
        .rolling(len(distances_1), min_periods=1)
        .min()
    )
    rolled_max_1 = (
        pd.Series(distances_1)
        .rolling(len(distances_1), min_periods=1)
        .max()
    )

    # normalise the time series values to lies within [0, 1]
    probability_of_crash_1 = (rolled_mean_1 - rolled_min_1) / (
        rolled_max_1 - rolled_min_1
    )

    # calculate rolling mean, min, max of homological derivatives
    rolled_mean_2 = pd.Series(distances_2).rolling(20, min_periods=1).mean()
    rolled_min_2 = (
        pd.Series(distances_2)
        .rolling(len(distances_2), min_periods=1)
        .min()
    )
    rolled_max_2 = (
        pd.Series(distances_2)
        .rolling(len(distances_2), min_periods=1)
        .max()
    )

    # normalise the time series values to lies within [0, 1]
    probability_of_crash_2 = (rolled_mean_2 - rolled_min_2) / (
        rolled_max_2 - rolled_min_2
    )

    # define time intervals to plots
    is_date_in_interval = (time_index_derivs > pd.Timestamp(start_date)) & (
        time_index_derivs < pd.Timestamp(end_date)
    )
    probability_of_crash_1_region = probability_of_crash_1[is_date_in_interval]
    probability_of_crash_2_region = probability_of_crash_2[is_date_in_interval]

    time_index_region = time_index_derivs[is_date_in_interval]
    resampled_close_price_region = price_resampled_derivs.loc[is_date_in_interval]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        resampled_close_price_region[probability_of_crash_1_region.values > threshold],
        '#ff7f0e', marker='.', linestyle='None', markersize=4
    )
    plt.plot(
        resampled_close_price_region[probability_of_crash_1_region.values <= threshold],
        "#1f77b4", marker='.', linestyle='None', markersize=4
    )
    plt.plot(
        resampled_close_price_region[probability_of_crash_1_region.values <= buy_threshold],
        "red", marker='.', linestyle='None', markersize=4
    )

    plt.title("Baseline Detector")
    plt.ylabel('Close Price', fontsize=12)
    plt.legend(
        [
            "Crash threshold > {0}".format(round(threshold, 3)),
            "Crash threshold ≤ {0}".format(round(threshold, 3)),
        ],
        loc="best",
        prop={"size": 10},
    )

    plt.subplot(1, 2, 2)
    plt.plot(
        resampled_close_price_region[probability_of_crash_2_region.values > threshold],
        '#ff7f0e', marker='.', linestyle='None', markersize=4
    )
    plt.plot(
        resampled_close_price_region[probability_of_crash_2_region.values <= threshold],
        "#1f77b4", marker='.', linestyle='None', markersize=4
    )
    plt.plot(
        resampled_close_price_region[probability_of_crash_2_region.values <= buy_threshold],
        "red", marker='.', linestyle='None', markersize=4
    )

    plt.title('Topological Detector')
    plt.legend(
        [
            "Crash threshold > {0}".format(round(threshold, 3)),
            "Crash threshold ≤ {0}".format(round(threshold, 3)),
        ],
        loc="best",
        prop={"size": 10},
    )

    plt.show()

def plot_crash_detections_std(
    start_date,
    end_date,
    distances,
    coefficient,
    time_index_derivs,
    price_resampled_derivs,
    metric_name
):

    # calculate rolling mean, min, max of homological derivatives
    rolled_mean_h = pd.Series(distances).rolling(20, min_periods=1).mean()
    rolled_min_h = (
        pd.Series(distances)
        .rolling(len(distances), min_periods=1)
        .min()
    )
    rolled_max_h = (
        pd.Series(distances)
        .rolling(len(distances), min_periods=1)
        .max()
    )

    # normalise the time series values to lies within [0, 1]
    probability_of_crash_h = (rolled_mean_h - rolled_min_h) / (
        rolled_max_h - rolled_min_h
    )

    # define time intervals to plots
    is_date_in_interval = (time_index_derivs > pd.Timestamp(start_date)) & (
        time_index_derivs < pd.Timestamp(end_date)
    )
    probability_of_crash_h_region = probability_of_crash_h[is_date_in_interval]
    time_index_region = time_index_derivs[is_date_in_interval]
    resampled_close_price_region = price_resampled_derivs.loc[is_date_in_interval]

    threshold = np.mean(probability_of_crash_h_region) + coefficient * np.std(probability_of_crash_h_region.values)
    buy_threshold = np.mean(probability_of_crash_h_region.values) - coefficient * np.std(probability_of_crash_h_region.values)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(time_index_region, probability_of_crash_h_region, color="#1f77b4")
    plt.axhline(y=threshold, linewidth=2, color='#ff7f0e', linestyle='--', label='Sell Threshold')
    plt.axhline(y=buy_threshold, linewidth=2, color='red', linestyle='--', label='Buy Threshold')
    plt.title(f"Crash Indicator Based on {metric_name}")
    plt.legend(loc="best", prop={"size": 10},)


    plt.subplot(1, 2, 2)
    plt.plot(
        resampled_close_price_region[probability_of_crash_h_region.values > threshold],
        '#ff7f0e', marker='.', linestyle='None', markersize=4
    )

    plt.plot(
        resampled_close_price_region[probability_of_crash_h_region.values <= threshold],
        color="#1f77b4", marker='.', linestyle='None', markersize=4
    )
    plt.plot(
        resampled_close_price_region[probability_of_crash_h_region.values <= buy_threshold],
        color='red', marker='.', linestyle='None', markersize=4
    )

    plt.title("Close Price")
    plt.legend(
        [
            "Crash threshold > {0}".format(round(threshold, 3)),
            "Crash threshold ≤ {0}".format(round(threshold, 3)),
            "Crash threshold ≤ {0}".format(round(buy_threshold, 3))
        ],
        loc="best",
        prop={"size": 10},
    )
    plt.savefig(fname=metric_name + ".png")
    plt.show()

def plot_crash_comparisons_std(
    start_date,
    end_date,
    coefficient,
    distances_1,
    distances_2,
    time_index_derivs,
    price_resampled_derivs
):

    # calculate rolling mean, min, max of homological derivatives
    rolled_mean_1 = pd.Series(distances_1).rolling(20, min_periods=1).mean()
    rolled_min_1 = (
        pd.Series(distances_1)
        .rolling(len(distances_1), min_periods=1)
        .min()
    )
    rolled_max_1 = (
        pd.Series(distances_1)
        .rolling(len(distances_1), min_periods=1)
        .max()
    )

    # normalise the time series values to lies within [0, 1]
    probability_of_crash_1 = (rolled_mean_1 - rolled_min_1) / (
        rolled_max_1 - rolled_min_1
    )

    # calculate rolling mean, min, max of homological derivatives
    rolled_mean_2 = pd.Series(distances_2).rolling(20, min_periods=1).mean()
    rolled_min_2 = (
        pd.Series(distances_2)
        .rolling(len(distances_2), min_periods=1)
        .min()
    )
    rolled_max_2 = (
        pd.Series(distances_2)
        .rolling(len(distances_2), min_periods=1)
        .max()
    )

    # normalise the time series values to lies within [0, 1]
    probability_of_crash_2 = (rolled_mean_2 - rolled_min_2) / (
        rolled_max_2 - rolled_min_2
    )

    # define time intervals to plots
    is_date_in_interval = (time_index_derivs > pd.Timestamp(start_date)) & (
        time_index_derivs < pd.Timestamp(end_date)
    )
    probability_of_crash_1_region = probability_of_crash_1[is_date_in_interval]
    probability_of_crash_2_region = probability_of_crash_2[is_date_in_interval]

    time_index_region = time_index_derivs[is_date_in_interval]
    resampled_close_price_region = price_resampled_derivs.loc[is_date_in_interval]

    threshold = np.mean(probability_of_crash_1_region) + coefficient * np.std(probability_of_crash_1_region.values)
    buy_threshold = np.mean(probability_of_crash_1_region.values) - coefficient * np.std(probability_of_crash_1_region.values)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        resampled_close_price_region[probability_of_crash_1_region.values > threshold],
        '#ff7f0e', marker='.', linestyle='None', markersize=4
    )
    plt.plot(
        resampled_close_price_region[probability_of_crash_1_region.values <= threshold],
        "#1f77b4", marker='.', linestyle='None', markersize=4
    )
    plt.plot(
        resampled_close_price_region[probability_of_crash_1_region.values <= buy_threshold],
        "red", marker='.', linestyle='None', markersize=4
    )

    plt.title("Baseline Detector")
    plt.ylabel('Close Price', fontsize=12)
    plt.legend(
        [
            "Crash threshold > {0}".format(round(threshold, 3)),
            "Crash threshold ≤ {0}".format(round(threshold, 3)),
        ],
        loc="best",
        prop={"size": 10},
    )

    threshold = np.mean(probability_of_crash_2_region) + coefficient * np.std(probability_of_crash_2_region.values)
    buy_threshold = np.mean(probability_of_crash_2_region.values) - coefficient * np.std(probability_of_crash_2_region.values)

    plt.subplot(1, 2, 2)
    plt.plot(
        resampled_close_price_region[probability_of_crash_2_region.values > threshold],
        '#ff7f0e', marker='.', linestyle='None', markersize=4
    )
    plt.plot(
        resampled_close_price_region[probability_of_crash_2_region.values <= threshold],
        "#1f77b4", marker='.', linestyle='None', markersize=4
    )
    plt.plot(
        resampled_close_price_region[probability_of_crash_2_region.values <= buy_threshold],
        "red", marker='.', linestyle='None', markersize=4
    )

    plt.title('Topological Detector')
    plt.legend(
        [
            "Crash threshold > {0}".format(round(threshold, 3)),
            "Crash threshold ≤ {0}".format(round(threshold, 3)),
        ],
        loc="best",
        prop={"size": 10},
    )

    plt.savefig(fname="Comparison Distances.png", format='png')
    plt.show()


def calc_returns(
        start_date,
        end_date,
        coefficient,
        distances,
        time_index_derivs,
        price_resampled_derivs,
        metric_name
):
    # calculate rolling mean, min, max of homological derivatives
    rolled_mean_h = pd.Series(distances).rolling(20, min_periods=1).mean()
    rolled_min_h = (
        pd.Series(distances)
            .rolling(len(distances), min_periods=1)
            .min()
    )
    rolled_max_h = (
        pd.Series(distances)
            .rolling(len(distances), min_periods=1)
            .max()
    )

    # normalise the time series values to lies within [0, 1]
    probability_of_crash_h = (rolled_mean_h - rolled_min_h) / (
            rolled_max_h - rolled_min_h
    )

    # define time intervals to plots
    is_date_in_interval = (time_index_derivs > pd.Timestamp(start_date)) & (
            time_index_derivs < pd.Timestamp(end_date)
    )
    probability_of_crash_h_region = probability_of_crash_h[is_date_in_interval]
    time_index_region = time_index_derivs[is_date_in_interval]
    resampled_close_price_region = price_resampled_derivs.loc[is_date_in_interval]

    threshold = np.mean(probability_of_crash_h_region) + coefficient * np.std(probability_of_crash_h_region.values)
    buy_threshold = np.mean(probability_of_crash_h_region.values) - coefficient * np.std(
        probability_of_crash_h_region.values)


    balance = 0
    units = 0
    interactions = 0
    # buying the first unit
    for i in range(len(resampled_close_price_region)):
        prob = probability_of_crash_h_region.values[i]
        price = resampled_close_price_region.values[i]

        if prob <= buy_threshold:
            units += 100000 / price
            break

    for i in range(len(resampled_close_price_region)):
        prob = probability_of_crash_h_region.values[i]
        new_price = resampled_close_price_region.values[i]

        if prob > threshold:
            balance += units * new_price
            units = 0
            interactions += 1

        if prob <= buy_threshold:
            units += balance/new_price
            balance = 0
            interactions += 1

    price = resampled_close_price_region.values[-1]
    balance += units * price
    return balance

def calc_returns_multiple_metrics(
        start_date,
        end_date,
        coefficient,
        distances_1,
        distances_2,
        time_index_derivs,
        price_resampled_derivs
):
    # calculate rolling mean, min, max of homological derivatives
    rolled_mean_1 = pd.Series(distances_1).rolling(20, min_periods=1).mean()
    rolled_min_1 = (
        pd.Series(distances_1)
            .rolling(len(distances_1), min_periods=1)
            .min()
    )
    rolled_max_1 = (
        pd.Series(distances_1)
            .rolling(len(distances_1), min_periods=1)
            .max()
    )

    # normalise the time series values to lies within [0, 1]
    probability_of_crash_1 = (rolled_mean_1 - rolled_min_1) / (
            rolled_max_1 - rolled_min_1
    )

    # calculate rolling mean, min, max of homological derivatives
    rolled_mean_2 = pd.Series(distances_2).rolling(20, min_periods=1).mean()
    rolled_min_2 = (
        pd.Series(distances_2)
            .rolling(len(distances_2), min_periods=1)
            .min()
    )
    rolled_max_2 = (
        pd.Series(distances_2)
            .rolling(len(distances_2), min_periods=1)
            .max()
    )

    # normalise the time series values to lies within [0, 1]
    probability_of_crash_2 = (rolled_mean_2 - rolled_min_2) / (
            rolled_max_2 - rolled_min_2
    )

    # define time intervals to plots
    is_date_in_interval = (time_index_derivs > pd.Timestamp(start_date)) & (
            time_index_derivs < pd.Timestamp(end_date)
    )
    probability_of_crash_1_region = probability_of_crash_1[is_date_in_interval]
    probability_of_crash_2_region = probability_of_crash_2[is_date_in_interval]

    time_index_region = time_index_derivs[is_date_in_interval]
    resampled_close_price_region = price_resampled_derivs.loc[is_date_in_interval]

    threshold1 = np.mean(probability_of_crash_1_region) + coefficient * np.std(probability_of_crash_1_region.values)
    buy_threshold1 = np.mean(probability_of_crash_1_region.values) - coefficient * np.std(
        probability_of_crash_1_region.values)

    threshold2 = np.mean(probability_of_crash_2_region) + coefficient * np.std(probability_of_crash_2_region.values)
    buy_threshold2 = np.mean(probability_of_crash_2_region.values) - coefficient * np.std(
        probability_of_crash_2_region.values)

    balance = 0
    units = 0
    # buying the first unit
    for i in range(len(resampled_close_price_region)):
        prob1 = probability_of_crash_1_region.values[i]
        prob2 = probability_of_crash_2_region.values[i]
        price = resampled_close_price_region.values[i]

        if prob1 <= buy_threshold1 and prob2 <= buy_threshold2:
            units = 100000 / price
            break


    for i in range(len(resampled_close_price_region)):
        prob1 = probability_of_crash_1_region.values[i]
        prob2 = probability_of_crash_2_region.values[i]
        new_price = resampled_close_price_region.values[i]

        if prob1 > threshold1 and prob2 > threshold2:
            balance += units * new_price
            units = 0

        if prob1 <= buy_threshold1 and prob2 <= buy_threshold2:
            units += balance/new_price
            balance = 0

    price = resampled_close_price_region.values[-1]
    balance += units * price
    return balance