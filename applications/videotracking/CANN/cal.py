# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import numpy as np
from matplotlib import pyplot as plt


def get_axis_aligned_bbox(region):
    region = np.asarray(region)
    nv = len(region)
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        return (cx - w / 2, cy - h / 2, w, h)
    else:
        return (region[0], region[1], region[2] - region[0], region[3] - region[1])


# Pass in the coordinates of the lower-left and upper-right corners of the two rectangles to derive the intersection and the union
def computeArea(rect1, rect2):
    # Keep rect 1 to the left.
    if rect1[0] > rect2[0]:
        return computeArea(rect2, rect1)
    # No overlap
    if rect1[1] >= rect2[3] or rect1[3] <= rect2[1] or rect1[2] <= rect2[0]:
        return 0, rect1[2] * rect1[3] + rect2[2] * rect2[3]
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])

    rect1w = rect1[2] - rect1[0]
    rect1h = rect1[3] - rect1[1]
    rect2w = rect2[2] - rect2[0]
    rect2h = rect2[3] - rect2[1]
    return abs(x1 - x2) * abs(y1 - y2), rect1w * rect1h + rect2w * rect2h - abs(x1 - x2) * abs(y1 - y2)


# Read coordinates from file
def readData(path, separator, need):
    reader = open(path, "r", encoding='utf-8')
    ans = []
    lines = reader.readlines()
    for i in range(len(lines)):
        t = lines[i].split(separator)
        t = [float(i) for i in t]
        if need:
            ans.append(get_axis_aligned_bbox(t))
        else:
            ans.append(t)
    return ans


def getCenter(region):
    return (region[0] + region[2] / 2, region[1] + region[3] / 2)


def computePrecision(myData, trueData, x):
    # Getting the center difference
    cen_gap = []
    for i in range(len(myData)):
        x1 = myData[i][0]
        y1 = myData[i][1]
        x2 = trueData[i][0]
        y2 = trueData[i][1]
        cen_gap.append(np.sqrt((x2-x1)**2+(y2-y1)**2))
    # Calculation of percentages
    precision = []
    for i in range(len(x)):
        gap = x[i]
        count = 0
        for j in range(len(cen_gap)):
            if cen_gap[j] < gap:
                count += 1
        precision.append(count/len(cen_gap))

    return precision


def computeSuccess(myData, trueData, x):
    frames = len(trueData)
    # Getting the Recombination Rate Score
    overlapScore = []

    for i in range(frames):
        one = [myData[i][0] - myData[i][2] / 2, myData[i][1] - myData[i][3] / 2,
               myData[i][0] + myData[i][2] / 2, myData[i][1] + myData[i][3] / 2]
        two = [trueData[i][0] - trueData[i][2]/2, trueData[i][1] - trueData[i][3]/2,
               trueData[i][0] + trueData[i][2]/2, trueData[i][1] + trueData[i][3]/2]
        a, b = computeArea(one, two)
        overlapScore.append(a / b)

    # Calculation of percentages
    success_rate = []
    for i in range(len(x)):
        gap = x[i]
        count = 0
        for j in range(frames):
            if overlapScore[j] > gap:
                count += 1
        success_rate.append(count/(frames-1))

    return success_rate


def showPrecision(myData, trueData, algorithm, colors):
    # Generate a threshold, calculated over the range [start, stop],
    # returning num (default 50) evenly spaced samples
    xPrecision = np.linspace(0, 10, 20)
    yPrecision = []
    for i in myData:
        # Store the horizontal and vertical coordinates of all the points.
        yPrecision.append(computePrecision(i, trueData, xPrecision))

    plt.figure('Precision plot in different algorithms')
    ax = plt.gca()
    ax.set_xlabel('Location error threshold')
    ax.set_ylabel('Precision')

    for i in range(len(myData)):
        # Draw a line graph with values in x_list as horizontal coordinates and values in y_list as vertical coordinates
        # The c parameter specifies the color of the line, the linewidth specifies the width of the line, 
        # and the alpha specifies the transparency of the line.
        ax.plot(xPrecision, yPrecision[i], color=colors[i], linewidth=1,
                alpha=0.6, label=algorithm[i] + "[%.3f]" % yPrecision[i][-1])

    # Best place to set the legend
    plt.legend(loc="best")
    plt.show()


def showSuccess(myData, trueData, algorithm, colors):
    # Generate a threshold, calculated over the range [start, stop],
    # returning num (default 50) evenly spaced samples
    xSuccess = np.linspace(0, 1, 20)

    # Store the horizontal and vertical coordinates of all the points.
    ySuccess = computeSuccess(myData, trueData, xSuccess)

    plt.figure('Success plot')
    ax = plt.gca()
    ax.set_xlabel('Overlap threshold')
    ax.set_ylabel('Success rate')

    auc = sum(ySuccess) / 20
    ax.plot(xSuccess, ySuccess, color=colors, linewidth=1,
            alpha=0.6, label="AUC" + "[%.3f]" % auc)

    plt.legend(loc="best")
    plt.savefig('success_plot.png')
    # plt.show()

