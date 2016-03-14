from __future__ import division
from matplotlib import pyplot as plt

def get_errors(arr):
    print len(arr)
    for i in xrange(1, 10):
        arr[i - 1] = i * 5 / arr[i - 1]



if __name__ == "__main__":
    x = [i*5 for i in xrange(1, 10)]
    print len(x)
    Y_1 = [5.151, 9.11, 12.60, 17.200, 6.811, 16.64, 16.07, 9.38, 42.07]
    Y_2 = [4.890, 9.4180, 13.3921, 17.373, 21.513, 29.5423, 15.09, 20.70, 12.5235]
    Y_3 = [4.560, 9.1028, 13.349, 18.246, 21.55, 10.808, 28.6230, 18.1100, 49.45]
    Y_4 = [4.356, 9.1109, 13.229, 18.134, 22.07, 28.98,  10.087, 10.08, 11.132]
    Y_5 = [5.023, 9.8771, 15.121, 18.134, 24.08, 29.09, 33.12, 12.32, 32.01]

    get_errors(Y_1)
    get_errors(Y_2)
    get_errors(Y_3)
    get_errors(Y_4)
    get_errors(Y_5)

    line1, = plt.plot(x, Y_1, label="image set 1")
    line2, = plt.plot(x, Y_2, label="image set 2")
    line3, = plt.plot(x, Y_3, label='image set 3')
    line4, = plt.plot(x, Y_4, label='image set 4')
    line5, = plt.plot(x, Y_5, label='image set 5')

    plt.title("Wheel Turning Angle versus Visual Angle Error")
    plt.legend([line1, line2, line3, line4, line5])

    plt.show()
