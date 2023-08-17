
import fileinput
from csv import DictReader
from collections import defaultdict
from matplotlib import pyplot as plt


def read_data():
    reader = DictReader(fileinput.input())
    rtn = defaultdict(lambda: defaultdict(list))
    for row in reader:
        # encountered another header row
        if row['h'] == 'h':
            continue
        # average over data points
        key = (float(row['h']), float(row['energy_point']))
        for k in ('entropy', 'ratio'):
            rtn[key][k].append(float(row[k]))

    # average
    for data_pt in rtn.values():
        for k, v in data_pt.items():
            data_pt[k] = sum(v)/len(v)

    return rtn


def plot_data(data):
    f, axes = plt.subplots(2, 1, sharex=True)

    for metric, ax in zip(['entropy', 'ratio'], axes):
        for energy_pt in sorted(set(e for _, e in data)):
            x, y = zip(*((h, v[metric]) for (h, e), v in data.items() if e == energy_pt))
            ax.plot(x, y)
            ax.set_ylabel(metric)

    plt.xlabel('Disorder strength $h$')

    plt.show()


def main():
    data = read_data()
    plot_data(data)


if __name__ == '__main__':
    main()
