import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(
    description="Analyze the time since starting an infretis simulation."
)

parser.add_argument("-f", help="The sim.log file to analyze")

args = parser.parse_args()

print('minutes since start')

def grep(infile):
    out = []
    with open(infile) as f:
        for line in f.readlines():
            if "date" in line:
                out.append(line.split()[-2:])
    return out


def calculate_minutes_since_start(datetime_array):
    # Extract dates and times from the input array
    dates = datetime_array[:, 0]
    times = datetime_array[:, 1]

    # Convert dates and times to datetime objects
    datetime_strings = [f"{date} {time}" for date, time in zip(dates, times)]
    datetimes = np.array(
        [datetime.strptime(dt, "%Y.%m.%d %H:%M:%S") for dt in datetime_strings]
    )

    # Calculate the time difference in minutes
    minutes_since_start = (datetimes - datetimes[0]).astype(
        "timedelta64[m]"
    ) / np.timedelta64(1, "m")

    return minutes_since_start


t = np.array(grep(args.f), dtype=str)
result = calculate_minutes_since_start(t)

plt.plot(result)
plt.ylabel("minutes since start")
plt.show()
