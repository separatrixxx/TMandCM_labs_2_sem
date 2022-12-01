import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import sympy as sp
import math

matplotlib.use('TkAgg')


def Spring(x0, y0, phi, rad):  # return lists for a spring
    SX = [x0 + rad * t * sp.cos(t) / (6 * math.pi) for t in np.linspace(0, 6 * math.pi + (1 / 2) * math.pi + phi, 100)]
    SY = [y0 + rad * t * sp.sin(t) / (6 * math.pi) for t in np.linspace(0, 6 * math.pi + (1 / 2) * math.pi + phi, 100)]
    return SX, SY


def Triangle():  # на чём стоит спираль
    TX = [-1, 1, 0, -1]
    TY = [-1.5, -1.5, 0, -1.5]
    return TX, TY


def Ro(phi, rad):  # это тот зелёный лучик
    RX = [0, -rad * sp.sin(phi)]
    RY = [0, rad * sp.cos(phi)]
    return RX, RY


def Rectangle(phi, Rb, Rt):
    KX = -Rb * sp.sin(phi + 3 * math.pi / 8)  # тут точки прямоугольника
    LX = -Rt * sp.sin(phi + math.pi / 4)
    NX = -Rb * sp.sin(phi - 3 * math.pi / 8)
    MX = -Rt * sp.sin(phi - math.pi / 4)
    KY = Rb * sp.cos(phi + 3 * math.pi / 8)
    NY = Rb * sp.cos(phi - 3 * math.pi / 8)
    LY = Rt * sp.cos(phi + math.pi / 4)
    MY = Rt * sp.cos(phi - math.pi / 4)
    RcX = [KX, LX, MX, NX, KX]
    RcY = [KY, LY, MY, NY, KY]
    return RcX, RcY


h = 6


def Square(phi, s):
    # сторона квадрата
    a = 1
    X1, Y1 = - math.sqrt(s ** 2 + h ** 2) * sp.sin(phi + sp.atan(s / h)), \
             math.sqrt(s ** 2 + h ** 2) * sp.cos(phi + sp.atan(s / h))
    X2, Y2 = - math.sqrt(s ** 2 + h ** 2) * sp.sin(phi + sp.atan(s / h)) - a * sp.sin(phi), \
             math.sqrt(s ** 2 + h ** 2) * sp.cos(phi + sp.atan(s / h)) + a * sp.cos(phi)
    X3, Y3 = - math.sqrt(s ** 2 + h ** 2) * sp.sin(phi + sp.atan(s / h)) - a * sp.sin(phi) - a * sp.cos(phi), \
             math.sqrt(s ** 2 + h ** 2) * sp.cos(phi + sp.atan(s / h)) + a * sp.cos(phi) - a * sp.sin(phi)
    X4, Y4 = - math.sqrt(s ** 2 + h ** 2) * sp.sin(phi + sp.atan(s / h)) - a * sp.cos(phi), \
             math.sqrt(s ** 2 + h ** 2) * sp.cos(phi + sp.atan(s / h)) - a * sp.sin(phi)

    RcX = [X1, X2, X3, X4, X1]
    RcY = [Y1, Y2, Y3, Y4, Y1]
    return RcX, RcY


# defining parameters
a = 10
ro = 2.5
Rtop = 6 * (2 ** 0.5)
Rbot = 3 * (5 ** 0.5)

# defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')
# here x, y, Vx, Vy, Wx, Wy, xC are functions of 't'
phi = 0.8 * sp.sin(4 * t)
s = 5 * sp.sin(4 * t)
# s = 5


# constructing corresponding arrays
T = np.linspace(0, 20, 1000)
XSpr = np.zeros_like(T)
YSpr = np.zeros_like(T)
Phi = np.zeros_like(T)
S = np.zeros_like(T)
# filling arrays with corresponding values
for i in np.arange(len(T)):
    Phi[i] = sp.Subs(phi, t, T[i])
    S[i] = sp.Subs(s, t, T[i])

# here we start to plot
fig = plt.figure(figsize=(17, 8))

ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')
ax1.set(xlim=[XSpr.min() - 2 * a, XSpr.max() + 2 * a], ylim=[YSpr.min() - 2 * a, YSpr.max() + 2 * a])

# plotting a spring
SpX, SpY = Spring(XSpr[0], YSpr[0], Phi[0], a / 8)
Spr, = ax1.plot(SpX, SpY, 'black')

# plotting triangle
TrX, TrY = Triangle()
Trngl, = ax1.plot(TrX, TrY, 'brown')

# plotting Ro
RoX, RoY = Ro(Phi[0], ro)
Rorad, = ax1.plot(RoX, RoY, 'green')

# plotting Rectangle
RectangleX, RectangleY = Rectangle(Phi[0], Rbot, Rtop)
Rect, = ax1.plot(RectangleX, RectangleY, 'purple')

# plotting Square
SquareX, SquareY = Square(Phi[0], S[0])
Sqr, = ax1.plot(SquareX, SquareY, 'red')

VmodA = sp.diff(phi, t)
WmodA = sp.diff(VmodA, t)

"""constructing functions"""
countOfFrames = 200
T_start, T_stop = 0, 12
T = np.linspace(T_start, T_stop, countOfFrames)

VmodA_def = sp.lambdify(t, VmodA)
WmodA_def = sp.lambdify(t, WmodA)

VA = VmodA_def(T)
WA = WmodA_def(T)

# plotting T-V and T-W
ax2 = fig.add_subplot(4, 2, 2)
ax2.set(xlim=[T_start, T_stop], ylim=[VA.min(), VA.max()])
tv_x = [T[0]]
tv_y = [VA[0]]
TV, = ax2.plot(tv_x, tv_y, '-')
ax2.set_xlabel('T')
ax2.set_ylabel('V')

ax3 = fig.add_subplot(4, 2, 4)
ax3.set(xlim=[T_start, T_stop], ylim=[WA.min(), WA.max()])
tw_x = [T[0]]
tw_y = [WA[0]]
TW, = ax3.plot(tv_x, tv_y, '-')

ax3.set_xlabel('T')
ax3.set_ylabel('W')

plt.subplots_adjust(wspace=0.3, hspace=0.7)


# function for recounting the positions


def anima(i):
    SpX, SpY = Spring(XSpr[i], YSpr[i], Phi[i], a / 8)
    Spr.set_data(SpX, SpY)
    RoX, RoY = Ro(Phi[i], ro)
    Rorad.set_data(RoX, RoY)
    RecX, RecY = Rectangle(Phi[i], Rbot, Rtop)
    Rect.set_data(RecX, RecY)
    SqX, SqY = Square(Phi[i], S[i])
    Sqr.set_data(SqX, SqY)

    tv_x.append(T[i])
    tv_y.append(VA[i])
    tw_x.append(T[i])
    tw_y.append(WA[i])
    TV.set_data(tv_x, tv_y)
    TW.set_data(tw_x, tw_y)
    if i == countOfFrames - 1:
        tv_x.clear()
        tv_y.clear()
        tw_x.clear()
        tw_y.clear()

    return Spr, Rorad, Rect, Sqr, TV, TW


# animation function


anim = FuncAnimation(fig, anima, frames=1000, interval=0.01, blit=True)

plt.show()
