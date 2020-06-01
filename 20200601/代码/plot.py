# encoding:utf-8
'''
绘制accuracy、val_acc、loss、val_loss折线
'''
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

# 绘制折线图
def plot(accuracy, val_acc, loss, val_loss):
    host = host_subplot(111)
    plt.grid(True) # 网格
    host.set_xlabel("steps")
    host.set_ylabel("rate")

    plt.axhline(y=1, ls=":", c="black") # 绘制y=1的直线
    p1, = host.plot(range(len(accuracy)), accuracy, label="accuracy")
    p2, = host.plot(range(len(val_acc)), val_acc, label="val_acc")
    p3, = host.plot(range(len(loss)), loss, label="loss")
    p4, = host.plot(range(len(val_loss)), val_loss, label="val_loss")
    host.legend(loc=6) # 图例位置，6位左侧

    plt.draw()
    plt.savefig('linechart.png')
    plt.show()

