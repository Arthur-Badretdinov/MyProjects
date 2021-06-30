import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D
from numpy.random import randn
from matplotlib.legend_handler import HandlerPatch

def example_N1_1():
    # В этом примере записываем значения y и автоматически генерируются значения x и рисуем график с синими линиями
    plt.plot([1, 3, 2])
    plt.ylabel('some numbers')
    plt.show()

def example_N1_2():
    # В этом примере записываем значения y и x и рисуем график с синими линиями
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.show()

def example_N1_3():
    # В этом примере записываем значения y и x, ставим диапозон для y и x и строим график с красными точками
    plt.plot([1,2,3,4], [1,4,9,16], "ro")
    plt.axis([0, 6, 0, 20])
    plt.show()

def example_N1_4():
    # Рассмотрим как будет меняться график в течении времени, для этого создается массив
    t = np.arange(0., 5., 0.2)

    # 3 графика: с красными линиями, синими квадратами и зелеными треугольниками
    plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
    plt.show()

def example_N1_5():
    # С помощью setp можно установить свойства для графика
    lines = plt.plot([1, 2, 3, 4], [1, 4, 9, 16], '-')
    plt.setp(lines, color='r', linewidth=4.0)
    plt.show()

def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)

def example_N1_6():
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)

    plt.figure(1) #Создаем первую фигуру
    plt.subplot(211) #Создаем первый подграфик для первой фигуры
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

    plt.subplot(212) #Создаем второй подграфик для первой фигуры
    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
    plt.show()

def example_N1_7():
    plt.figure(1)
    plt.subplot(211)
    plt.plot([1, 2, 3])
    plt.subplot(212)
    plt.plot([4, 5, 6])

    plt.figure(2)
    plt.plot([4, 5, 6])

    plt.figure(1)
    plt.subplot(211)
    plt.title('Easy as 1, 2, 3')
    plt.show()

def example_N1_8():
    np.random.seed(19680801)

    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

    plt.xlabel('Smarts') #Пишем название оси x
    plt.ylabel('Probability') #Пишем название оси y
    plt.title('Histogram of IQ') #Пишем название граика
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')  # Ставим надпись в графике
    plt.axis([40, 160, 0, 0.03]) #Устанавливаем границы осей
    plt.grid(True)
    plt.show()

def example_N1_9():
    ax = plt.subplot(111)

    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2 * np.pi * t)
    line, = plt.plot(t, s, lw=2)

    plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )

    plt.ylim(-2, 2)
    plt.show()

def example_N1_10():
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # make up some data in the interval ]0, 1[
    y = np.random.normal(loc=0.5, scale=0.4, size=1000)
    y = y[(y > 0) & (y < 1)]
    y.sort()
    x = np.arange(len(y))

    # plot with various axes scales
    plt.figure(1)

    # linear
    plt.subplot(221)
    plt.plot(x, y)
    plt.yscale('linear')
    plt.title('linear')
    plt.grid(True)

    # log
    plt.subplot(222)
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('log')
    plt.grid(True)

    # symmetric log
    plt.subplot(223)
    plt.plot(x, y - y.mean())
    plt.yscale('symlog', linthreshy=0.01)
    plt.title('symlog')
    plt.grid(True)

    # logit
    plt.subplot(224)
    plt.plot(x, y)
    plt.yscale('logit')
    plt.title('logit')
    plt.grid(True)
    # Format the minor tick labels of the y-axis into empty strings with
    # `NullFormatter`, to avoid cumbering the axis with too many labels.
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    # Adjust the subplot layout, because the logit one may take more space
    # than usual, due to y-tick labels like "1 - 10^{-3}"
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.35)
    plt.show()


def example_N2_1():
    n = 256
    X = np.linspace(-np.pi, np.pi, n, endpoint=True)
    Y = np.sin(2 * X)

    plt.axes([0.025, 0.025, 0.95, 0.95])

    plt.plot(X, Y + 1, color='blue', alpha=1.00)
    plt.fill_between(X, 1, Y + 1, color='blue', alpha=.25)

    plt.plot(X, Y - 1, color='blue', alpha=1.00)
    plt.fill_between(X, -1, Y - 1, (Y - 1) > -1, color='blue', alpha=.25)
    plt.fill_between(X, -1, Y - 1, (Y - 1) < -1, color='red', alpha=.25)

    plt.xlim(-np.pi, np.pi), plt.xticks([])
    plt.ylim(-2.5, 2.5), plt.yticks([])
    plt.show()

def example_N2_2():
    n = 1024
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X)

    plt.axes([0.025, 0.025, 0.95, 0.95])
    plt.scatter(X, Y, s=75, c=T, alpha=.5)

    plt.xlim(-1.5, 1.5), plt.xticks([])
    plt.ylim(-1.5, 1.5), plt.yticks([])
    plt.show()

def example_N2_3():
    n = 12
    X = np.arange(n)
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

    plt.axes([0.025, 0.025, 0.95, 0.95])
    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

    for x, y in zip(X, Y1):
        plt.text(x , y + 0.05, '%.2f' % y, ha='center', va='bottom')

    for x, y in zip(X, Y2):
        plt.text(x, -y - 0.05, '%.2f' % y, ha='center', va='top')

    plt.xlim(-.5, n), plt.xticks([])
    plt.ylim(-1.25, +1.25), plt.yticks([])
    plt.show()

def f1(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

def example_N2_4():
    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(x, y)

    plt.axes([0.025, 0.025, 0.95, 0.95])

    plt.contourf(X, Y, f1(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
    C = plt.contour(X, Y, f1(X, Y), 8, colors='black', linewidth=.5)
    plt.clabel(C, inline=1, fontsize=10)

    plt.xticks([]), plt.yticks([])
    plt.show()

def example_N2_5():
    n = 10
    x = np.linspace(-3, 3, 35)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    Z = f1(X, Y)

    plt.axes([0.025, 0.025, 0.95, 0.95])
    plt.imshow(Z, interpolation='bicubic', cmap='bone', origin='lower')
    plt.colorbar(shrink=.92)

    plt.xticks([]), plt.yticks([])
    plt.show()

def example_N2_6():
    n = 20
    Z = np.ones(n)
    Z[-1] *= 2

    plt.axes([0.025, 0.025, 0.95, 0.95])

    plt.pie(Z, explode=Z * .05, colors=['%f' % (i / float(n)) for i in range(n)],
            wedgeprops={"linewidth": 1, "edgecolor": "black"})
    plt.gca().set_aspect('equal')
    plt.xticks([]), plt.yticks([])
    plt.show()

def example_N2_7():
    n = 8
    X, Y = np.mgrid[0:n, 0:n]
    T = np.arctan2(Y - n / 2.0, X - n / 2.0)
    R = 10 + np.sqrt((Y - n / 2.0) ** 2 + (X - n / 2.0) ** 2)
    U, V = R * np.cos(T), R * np.sin(T)

    plt.axes([0.025, 0.025, 0.95, 0.95])
    plt.quiver(X, Y, U, V, R, alpha=.5)
    plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)

    plt.xlim(-1, n), plt.xticks([])
    plt.ylim(-1, n), plt.yticks([])
    plt.show()

def example_N2_8():
    ax = plt.axes([0.025, 0.025, 0.95, 0.95])

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
    ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.show()

def example_N2_9():
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.025, left=0.025, top=0.975, right=0.975)

    plt.subplot(2, 1, 1)
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 4)
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 5)
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 6)
    plt.xticks([]), plt.yticks([])

    plt.show()

def example_N2_10():
    ax = plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)

    N = 20
    theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
    radii = 10 * np.random.rand(N)
    width = np.pi / 4 * np.random.rand(N)
    bars = plt.bar(theta, radii, width=width, bottom=0.0)

    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        bar.set_alpha(0.5)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()

def example_N2_11():
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_zlim(-2, 2)

    # savefig('../figures/plot3d_ex.png',dpi=48)
    plt.show()

def example_N2_12():
    eqs = []
    eqs.append((
                   r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} \int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$"))
    eqs.append(
        (r"$\frac{d\rho}{d t} + \rho \vec{v}\cdot\nabla\vec{v} = -\nabla p + \mu\nabla^2 \vec{v} + \rho \vec{g}$"))
    eqs.append((r"$\int_{-\infty}^\infty e^{-x^2}dx=\sqrt{\pi}$"))
    eqs.append((r"$E = mc^2 = \sqrt{{m_0}^2c^4 + p^2c^2}$"))
    eqs.append((r"$F_G = G\frac{m_1m_2}{r^2}$"))

    plt.axes([0.025, 0.025, 0.95, 0.95])

    for i in range(24):
        index = np.random.randint(0, len(eqs))
        eq = eqs[index]
        size = np.random.uniform(12, 32)
        x, y = np.random.uniform(0, 1, 2)
        alpha = np.random.uniform(0.25, .75)
        plt.text(x, y, eq, ha='center', va='center', color="#11557c", alpha=alpha,
                 transform=plt.gca().transAxes, fontsize=size, clip_on=True)

    plt.xticks([]), plt.yticks([])
    plt.show()

def example_N3():
    fig = plt.figure()
    fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title('axes title')

    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')

    ax.text(3, 8, 'boxed italics text in data coords', style='italic',
            bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

    ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

    ax.text(3, 2, u'unicode: Institut f\374r Festk\366rperphysik')

    ax.text(0.95, 0.01, 'colored text in axes coords',
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='green', fontsize=15)

    ax.plot([2], [1], 'o')
    ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
                arrowprops=dict(facecolor='black', shrink=0.05))

    ax.axis([0, 10, 0, 10])

    plt.show()

def example_N4():
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2 * np.pi * t)

    plt.plot(t, s)
    plt.title(r'$\alpha_i > \beta_i$', fontsize=20)
    plt.text(1, -0.6, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
    plt.text(0.6, 0.6, r'$\mathcal{A}\mathrm{sin}(2 \omega t)$',
             fontsize=20)
    plt.xlabel('time (s)')
    plt.ylabel('volts (mV)')
    plt.show()

def example_N5_1():
    red_patch = mpatches.Patch(color='red', label='The red data')
    plt.legend(handles=[red_patch])

    plt.show()

def example_N5_2():
    blue_line = mlines.Line2D([], [], color='blue', marker='*',
                              markersize=15, label='Blue stars')
    plt.legend(handles=[blue_line])

    plt.show()

def example_N5_3():
    plt.subplot(211)
    plt.plot([1, 2, 3], label="test1")
    plt.plot([3, 2, 1], label="test2")
    # Place a legend above this subplot, expanding itself to
    # fully use the given bounding box.
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.subplot(223)
    plt.plot([1, 2, 3], label="test1")
    plt.plot([3, 2, 1], label="test2")
    # Place a legend to the right of this smaller subplot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

def example_N5_4():
    line1, = plt.plot([1, 2, 3], label="Line 1", linestyle='--')
    line2, = plt.plot([3, 2, 1], label="Line 2", linewidth=4)

    first_legend = plt.legend(handles=[line1], loc=1)

    ax = plt.gca().add_artist(first_legend)

    plt.legend(handles=[line2], loc=4)
    plt.show()

def example_N5_5():
    line1, = plt.plot([3, 2, 1], marker='o', label='Line 1')
    line2, = plt.plot([1, 2, 3], marker='o', label='Line 2')

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    plt.show()

def example_N5_6():
    z = randn(10)

    red_dot, = plt.plot(z, "ro", markersize=15)
    # Put a white cross over some of the data.
    white_cross, = plt.plot(z[:5], "w+", markeredgewidth=3, markersize=15)

    plt.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])
    plt.show()

class AnyObject(object):
    pass

class AnyObjectHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
                                   edgecolor='black', hatch='xx', lw=3,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch

def example_N5_7():
    plt.legend([AnyObject()], ['My first handler'],
               handler_map={AnyObject: AnyObjectHandler()})
    plt.show()

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def example_N5_8():
    c = mpatches.Circle((0.5, 0.5), 0.25, facecolor="green",
                        edgecolor="red", linewidth=3)
    plt.gca().add_patch(c)

    plt.legend([c], ["An ellipse, not a rectangle"],
               handler_map={mpatches.Circle: HandlerEllipse()})
    plt.show()

if __name__ == '__main__':
    #example_N1_1()
    #example_N1_2()
    #example_N1_3()
    #example_N1_4()
    #example_N1_5()
    #example_N1_6()
    #example_N1_7()
    #example_N1_8() #где-то тут ошибка
    #example_N1_9()
    #example_N1_10()
    #example_N2_1()
    #example_N2_2()
    #example_N2_3()
    #example_N2_4()
    #example_N2_5()
    #example_N2_6()
    #example_N2_7()
    #example_N2_8()
    #example_N2_9()
    #example_N2_10()
    #example_N2_11()
    #example_N2_12()
    #example_N3()
    #example_N4()
    #example_N5_1()
    #example_N5_2()
    #example_N5_3()
    #example_N5_4()
    #example_N5_5()
    #example_N5_6()
    #example_N5_7()
    example_N5_8()

