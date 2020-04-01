import random
import numpy as np
import copy
from itertools import product, combinations
import math

x1min = -5
x1max = 15
x2min = -25
x2max = 10
x3min = 15
x3max = 45
x01 = (x1max + x1min) / 2
x02 = (x2max + x2min) / 2
x03 = (x3max + x3min) / 2

xSrmax = x1max + x2max + x3max / 3
xSrmin = x1min + x2min + x3min / 3
gt = {1: 0.6798, 2: 0.5157, 3: 0.4377, 4: 0.3910, 5: 0.3595, 6: 0.3362, 7: 0.3185, 8: 0.3043, 9: 0.2926, 10: 0.2829}
print("\nРівняння регресії ПФЕ")
print("y=b0+b1*x1+b2*x2+b3*x3+b12*x1*x2+b13*x1*x3+b23*x2*x3+b123*x1*x2*x3+b11*x1^2+b22*x2^2+b33*x3^2\n")


def sumkf2(x1, x2):
    xn = []
    for i in range(len(x1)):
        xn.append(x1[i] * x2[i])
    return xn


def sumkf3(x1, x2, x3):
    xn = []
    for i in range(len(x1)):
        xn.append(x1[i] * x2[i] * x3[i])
    return xn


def kv(x):
    xn = []
    for i in range(len(x)):
        xn.append(x[i] * x[i])
    return xn


print("Матриця планування для повного трьохфакторного експерименту")
print(
    "{:3} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} ".format("№", "X0", "X1", "X2", "X3", "X4", "X5", "X6",
                                                                          "X7", "X8", "X9", "X10", ))
X1 = [x1min, x1min, x1min, x1min, x1max, x1max, x1max, x1max]
X2 = [x2min, x2min, x2max, x2max, x2min, x2min, x2max, x2max]
X3 = [x3min, x3max, x3min, x3max, x3min, x3max, x3min, x3max]
Y1 = [random.randrange(175, 261, 1) for i in range(8)]
Y2 = [random.randrange(175, 261, 1) for i in range(8)]
Y3 = [random.randrange(175, 261, 1) for i in range(8)]
X12 = sumkf2(X1, X2)
X13 = sumkf2(X1, X3)
X23 = sumkf2(X2, X3)
X123 = sumkf3(X1, X2, X3)
X0 = [1] * 8
Xall = [X0, X1, X2, X3, X12, X13, X23, X123]

ymax = int(200 + xSrmax)
ymin = int(200 + xSrmin)
for i in range(8):
    print(
        "{:1} {:5} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} {:6} ".format(i + 1, X0[i], X1[i], X2[i], X3[i], X12[i], X13[i], X23[i], X123[i], Y1[i], Y2[i], Y3[i]))

def make_norm_plan_matrix(plan_matrix, matrix_of_min_and_max_x):
    X0 = np.mean(matrix_with_min_max_x, axis=1)
    interval_of_change = np.array([(matrix_of_min_and_max_x[i, 1] - X0[i]) for i in range(len(plan_matrix[0]))])
    X_norm = np.array(
        [[round((plan_matrix[i, j] - X0[j]) / interval_of_change[j], 3) for j in range(len(plan_matrix[i]))]
         for i in range(len(plan_matrix))])
    return X_norm

matrix_with_min_max_x = np.array([[-5, 15], [-25, 10], [15, 45]])
m = 6
N = 8
norm_matrix = np.array(list(product("01", repeat=3)), dtype=np.int)
norm_matrix[norm_matrix == 0] = -1
norm_matrix = np.insert(norm_matrix, 0, 1, axis=1)
plan_matrix = np.empty((8, 3))
for i in range(len(norm_matrix)):
    for j in range(1, len(norm_matrix[i])):
        if j == 1:
            if norm_matrix[i, j] == -1:
                plan_matrix[i, j - 1] = -20
            elif norm_matrix[i, j] == 1:
                plan_matrix[i, j - 1] = 15
        elif j == 2:
            if norm_matrix[i, j] == -1:
                plan_matrix[i, j - 1] = 10
            elif norm_matrix[i, j] == 1:
                plan_matrix[i, j - 1] = 60
        elif j == 3:
            if norm_matrix[i, j] == -1:
                plan_matrix[i, j - 1] = 15
            elif norm_matrix[i, j] == 1:
                plan_matrix[i, j - 1] = 35
plan_matr = np.insert(plan_matrix, 0, 1, axis=1)
Y_matrix = np.random.randint(200 + np.mean(matrix_with_min_max_x, axis=0)[0],
                             200 + np.mean(matrix_with_min_max_x, axis=0)[1], size=(N, m))
mean_Y = np.mean(Y_matrix, axis=1)
combination = list(combinations(range(1, 4), 2))
for i in combination:
    plan_matr = np.append(plan_matr, np.reshape(plan_matr[:, i[0]] * plan_matr[:, i[1]], (8, 1)), axis=1)
    norm_matrix = np.append(norm_matrix, np.reshape(norm_matrix[:, i[0]] * norm_matrix[:, i[1]], (8, 1)), axis=1)
    norm_matrix = np.append(norm_matrix, np.reshape(norm_matrix[:, 1] * norm_matrix[:, 2] * norm_matrix[:, 3], (8, 1)),
                            axis=1)
    plan_matr = np.append(plan_matr, np.reshape(plan_matr[:, 1] * plan_matr[:, 2] * plan_matr[:, 3], (8, 1)), axis=1)
plan_matr = np.insert(plan_matrix, 0, 1, axis=1)

print("Нормована матриця: \n", norm_matrix)
print("Матриця відгуків: \n", Y_matrix)


def yav(i):
    return (Y1[i - 1] + Y2[i - 1] + Y3[i - 1]) / 3


ySr1 = yav(1)
ySr2 = yav(2)
ySr3 = yav(3)
ySr4 = yav(4)
ySr5 = yav(5)
ySr6 = yav(6)
ySr7 = yav(7)
ySr8 = yav(8)

mx1 = (x1min + x1min + x1max + x1max) / 4
mx2 = (x2min + x2max + x2min + x2max) / 4
mx3 = (x3min + x3max + x3max + x3min) / 4

my = (ySr1 + ySr2 + ySr3 + ySr4) / 4
print("\nКоефіціенти m:\n" "mx1=", mx1, "\nmx2=", mx2, "\nmx3=", mx3, "\nmy=", my, "\n")
a1 = (X1[0] * ySr1 + X1[1] * ySr2 + X1[2] * ySr3 + X1[3] * ySr4) / 4
a2 = (X2[0] * ySr1 + X2[1] * ySr2 + X2[2] * ySr3 + X2[3] * ySr4) / 4
a3 = (X3[0] * ySr1 + X3[1] * ySr2 + X3[2] * ySr3 + X3[3] * ySr4) / 4

a11 = (X1[0] * X1[0] + X1[1] * X1[1] + X1[2] * X1[2] + X1[3] * X1[3]) / 4
a22 = (X2[0] * X2[0] + X2[1] * X2[1] + X2[2] * X2[2] + X2[3] * X2[3]) / 4
a33 = (X3[0] * X3[0] + X3[1] * X3[1] + X3[2] * X3[2] + X3[3] * X3[3]) / 4
a12 = a21 = (X1[0] * X2[0] + X1[1] * X2[1] + X1[2] * X2[2] + X1[3] * X2[3]) / 4
a13 = a31 = (X1[0] * X3[0] + X1[1] * X3[1] + X1[2] * X3[2] + X1[3] * X3[3]) / 4
a23 = a32 = (X2[0] * X3[0] + X2[1] * X3[1] + X2[2] * X3[2] + X2[3] * X3[3]) / 4

print("Коефіціенти а:\n" "a1=", a1, "\na2=", a2, "\na3=", a3, "\na11=", a11, "\na22=", a22, "\na33=", a33, "\na12=",
      a12, "\na13=", a13, "\na23=", a23)

b01 = np.array([[my, mx1, mx2, mx3], [a1, a11, a12, a13], [a2, a12, a22, a32], [a3, a13, a23, a33]])
b02 = np.array([[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a32], [mx3, a13, a23, a33]])
b0 = np.linalg.det(b01) / np.linalg.det(b02)

b11 = np.array([[1, my, mx2, mx3], [mx1, a1, a12, a13], [mx2, a2, a22, a32], [mx3, a3, a23, a33]])
b12 = np.array([[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a32], [mx3, a13, a23, a33]])
b1 = np.linalg.det(b11) / np.linalg.det(b12)

b21 = np.array([[1, mx1, my, mx3], [mx1, a11, a1, a13], [mx2, a12, a2, a32], [mx3, a13, a3, a33]])
b22 = np.array([[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a32], [mx3, a13, a23, a33]])
b2 = np.linalg.det(b21) / np.linalg.det(b22)

b31 = np.array([[1, mx1, mx2, my], [mx1, a11, a12, a1], [mx2, a12, a22, a2], [mx3, a13, a23, a3]])
b32 = np.array([[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a32], [mx3, a13, a23, a33]])
b3 = np.linalg.det(b31) / np.linalg.det(b32)
print("\nОтримані детермінанти:", "\nb0=", b0, "\nb1=", b1, "\nb2=", b2, "\nb3=", b3)
print("\nОтримане рівняння регресії:" "\n" "y=", b0, "+", "(", b1, ")", "*x1", "+", "(", b2, ")", "*x2", "+", "(", b3,
      ")", "*x3")


def cochran_check(Y_matrix):
    mean_Y = np.mean(Y_matrix, axis=1)
    dispersion_Y = np.mean((Y_matrix.T - mean_Y) ** 2, axis=0)
    Gp = np.max(dispersion_Y) / (np.sum(dispersion_Y))
    return Gp < gt[m - 1]


print("\nПеревірка однорідності дисперсії за критерієм Кохрена:")

d1 = ((Y1[0] - ySr1) ** 2 + (Y2[0] - ySr2) ** 2 + (Y3[0] - ySr3) ** 2) / 3
d2 = ((Y1[1] - ySr1) ** 2 + (Y2[1] - ySr2) ** 2 + (Y3[1] - ySr3) ** 2) / 3
d3 = ((Y1[2] - ySr1) ** 2 + (Y2[2] - ySr2) ** 2 + (Y3[2] - ySr3) ** 2) / 3
d4 = ((Y1[3] - ySr1) ** 2 + (Y2[3] - ySr2) ** 2 + (Y3[3] - ySr3) ** 2) / 3
print("d1=", round(d1, 2), "d2=", round(d2, 2), "d3=", round(d3, 2), "d4=", round(d4, 2))

dcouple = [d1, d2, d3, d4]

m = 3
Gp = max(dcouple) / sum(dcouple)
f1 = m - 1
f2 = N = 4
Gt = 0.5679
print(Gp)
for number in range(1):
    if Gp < Gt:
        print("Дисперсія  однорідна")
    else:
        print("\nДисперсія неоднорідна\n")
print("\nЗначимість коефіцієнтів регресії згідно критерію Стьюдента:\n")
sb = sum(dcouple) / N
sb2 = sb / N * m
sbs = math.sqrt(sb2)

beta0 = (ySr1 * 1 + ySr2 * 1 + ySr3 * 1 + ySr4 * 1) / 4
beta1 = (ySr1 * (-1) + ySr2 * (-1) + ySr3 * 1 + ySr4 * 1) / 4
beta2 = (ySr1 * (-1) + ySr2 * 1 + ySr3 * (-1) + ySr4 * 1) / 4
beta3 = (ySr1 * (-1) + ySr2 * 1 + ySr3 * 1 + ySr4 * (-1)) / 4

t0 = abs(beta0) / sbs
t1 = abs(beta1) / sbs
t2 = abs(beta2) / sbs
t3 = abs(beta3) / sbs

f3 = f1 * f2
ttabl = 2.306

if (t0 < ttabl):
    print("t0<ttabl, b0 не значимий")
    b0 = 0
if (t1 < ttabl):
    print("t1<ttabl, b1 не значимий")
    b1 = 0
if (t2 < ttabl):
    print("t2<ttabl, b2 не значимий")
    b2 = 0
if (t3 < ttabl):
    print("t3<ttabl, b3 не значимий\n")
    b3 = 0

yy1 = b0 + b1 * x1min + b2 * x2min + b3 * x3min
yy2 = b0 + b1 * x1min + b2 * x2max + b3 * x3max
yy3 = b0 + b1 * x1max + b2 * x2min + b3 * x3max
yy4 = b0 + b1 * x1max + b2 * x2max + b3 * x3min
print("Критерій Фішера\n")
d = 2
print(d, " значимих коефіцієнтів")
f4 = N - d

sad = ((yy1 - ySr1) ** 2 + (yy2 - ySr2) ** 2 + (yy3 - ySr3) ** 2 + (yy4 - ySr4) ** 2) * (m / (N - d))
Fp = sad / sb
print("d1=", round(d1, 2), "d2=", round(d2, 2), "d3=", round(d3, 2), "d4=", round(d4, 2), "d5=", round(sb, 2))
print("Fp=", round(Fp, 2))

Ft = 4.5
cont = 0
if Fp > Ft:
    print("Fp=", round(Fp, 2), ">Ft", Ft, "Отже, рівняння неадекватно оригіналу\n")
    cont = 1
else:
    print("Fp=", round(Fp, 2), "<Ft", Ft, "Отже, рівняння адекватно оригіналу\n")
if cont == 1:
    print("З взаємодією\n")
    Yall = [ySr1, ySr2, ySr3, ySr4, ySr5, ySr6, ySr7, ySr8]

    my = (ySr1 + ySr2 + ySr3 + ySr4 + ySr5 + ySr6 + ySr7 + ySr8) / 8

    def mni(j):
        s = 0
        for i in range(len(Xall[j])):
            s = s + Xall[j][i]
        return s / len(Xall[j])
    def mni2(j, k):
        s = 0
        for i in range(len(Xall[j])):
            s = s + Xall[j][i] * Xall[k][i]
        return s / len(Xall[j])
    def mni3(j, k, l):
        s = 0
        for i in range(len(Xall[j])):
            s = s + Xall[j][i] * Xall[k][i] * Xall[l][i]
        return s / len(Xall[j])
    def mni4(j1, j2, j3, j4):
        s = 0
        for i in range(len(Xall[j1])):
            s = s + Xall[j1][i] * Xall[j2][i] * Xall[j3][i] * Xall[j4][i]
        return s / len(Xall[j1])
    def mni5(j1, j2, j3, j4, j5):
        s = 0
        for i in range(len(Xall[j1])):
            s = s + Xall[j1][i] * Xall[j2][i] * Xall[j3][i] * Xall[j4][i] * Xall[j5][i]
        return s / len(Xall[j1])
    def mni6(j1, j2, j3, j4, j5, j6):
        s = 0
        for i in range(len(Xall[j1])):
            s = s + Xall[j1][i] * Xall[j2][i] * Xall[j3][i] * Xall[j4][i] * Xall[j5][i] * Xall[j6][i]
        return s / len(Xall[j1])
    Yalls = sum(Yall) / 8

    def kmn(j):
        s = 0
        for i in range(len(Yall)):
            s = s + yav(i) * Xall[j][i]
        return s / len(Yall)
    def kmn2(j, k):
        s = 0
        for i in range(len(Yall)):
            s = s + yav(i) * Xall[j][i] * Xall[k][i]
        return s / len(Yall)
    def kmn3(j, k, l):
        s = 0
        for i in range(len(Yall)):
            s = s + yav(i) * Xall[j][i] * Xall[k][i] * Xall[l][i]
        return s / len(Yall)

    if cochran_check(Y_matrix):
        b_natura = np.linalg.lstsq(plan_matr, mean_Y, rcond=None)[0]
        b_norm = np.linalg.lstsq(norm_matrix, mean_Y, rcond=None)[0]
        check1 = np.sum(b_natura * plan_matr, axis=1)
        check2 = np.sum(b_norm * norm_matrix, axis=1)
    print("Перевірка 1: ", check1)
    print("Перевірка 2: ", check2)
    M, P = 8, 8
    m = [[0 for y in range(M)] for x in range(P)]
    m[0][0] = 1
    m[1][0] = m[0][1] = mni(1)
    m[2][0] = m[0][2] = mni(2)
    m[3][0] = m[0][3] = mni(3)
    m[4][0] = m[0][4] = mni2(1, 2)
    m[5][0] = m[0][5] = mni2(1, 3)
    m[6][0] = m[0][6] = mni2(2, 3)
    m[7][0] = m[0][7] = m[1][6] = m[2][5] = m[3][4] = m[4][3] = m[5][2] = m[6][1] = mni3(1, 2, 3)
    m[1][1] = mni2(1, 1)
    m[1][2] = m[2][1] = mni2(1, 2)
    m[1][3] = m[3][1] = mni2(1, 3)
    m[1][4] = m[4][1] = mni3(1, 1, 2)
    m[1][5] = m[5][1] = mni3(1, 1, 3)
    m[1][7] = m[7][1] = mni4(1, 1, 2, 3)
    m[2][2] = mni2(2, 2)
    m[2][3] = m[3][2] = mni2(2, 3)
    m[2][4] = m[4][2] = mni3(1, 2, 2)
    m[2][6] = m[6][2] = mni3(2, 2, 3)
    m[2][7] = m[7][2] = mni4(1, 2, 2, 3)
    m[3][3] = mni2(3, 3)
    m[3][5] = m[5][3] = mni3(1, 3, 3)
    m[3][6] = m[6][3] = mni3(2, 3, 3)
    m[3][7] = m[7][3] = mni4(1, 2, 3, 3)
    m[4][4] = mni4(1, 1, 2, 2)
    m[4][5] = m[5][4] = mni4(1, 1, 2, 3)
    m[4][6] = m[6][4] = mni4(1, 2, 2, 3)
    m[4][7] = m[7][4] = mni5(1, 1, 2, 2, 3)
    m[5][5] = mni4(1, 1, 3, 3)
    m[5][6] = m[6][5] = mni4(1, 2, 3, 3)
    m[5][7] = m[7][5] = mni5(1, 1, 2, 3, 3)
    m[6][6] = mni4(2, 2, 3, 3)
    m[6][7] = m[7][6] = mni5(1, 2, 2, 3, 3)
    m[7][7] = mni6(1, 1, 2, 2, 3, 3)
    # print(m)

    # print(np.linalg.det(m))
    k0 = Yalls
    k1 = kmn(1)
    k2 = kmn(2)
    k3 = kmn(3)
    k4 = kmn2(1, 2)
    k5 = kmn2(1, 3)
    k6 = kmn2(2, 3)
    k7 = kmn3(1, 2, 3)
    k = [k0, k1, k2, k3, k4, k5, k6, k7]

    b00 = copy.deepcopy(m)
    b01 = copy.deepcopy(m)
    b02 = copy.deepcopy(m)
    b03 = copy.deepcopy(m)
    b04 = copy.deepcopy(m)
    b05 = copy.deepcopy(m)
    b06 = copy.deepcopy(m)
    b07 = copy.deepcopy(m)
    b00[0] = k
    b01[1] = k
    b02[2] = k
    b03[3] = k
    b04[4] = k
    b05[5] = k
    b06[6] = k
    b07[7] = k

    b0 = np.linalg.det(b00) / np.linalg.det(m)
    b1 = np.linalg.det(b01) / np.linalg.det(m)
    b2 = np.linalg.det(b02) / np.linalg.det(m)
    b3 = np.linalg.det(b03) / np.linalg.det(m)
    b4 = np.linalg.det(b04) / np.linalg.det(m)
    b5 = np.linalg.det(b05) / np.linalg.det(m)
    b6 = np.linalg.det(b06) / np.linalg.det(m)
    b7 = np.linalg.det(b07) / np.linalg.det(m)

    print("\nДисперсія по рядкам")
    d1 = ((Y1[0] - ySr1) ** 2 + (Y2[0] - ySr2) ** 2 + (Y3[0] - ySr3) ** 2) / 3
    d2 = ((Y1[1] - ySr1) ** 2 + (Y2[1] - ySr2) ** 2 + (Y3[1] - ySr3) ** 2) / 3
    d3 = ((Y1[2] - ySr1) ** 2 + (Y2[2] - ySr2) ** 2 + (Y3[2] - ySr3) ** 2) / 3
    d4 = ((Y1[3] - ySr1) ** 2 + (Y2[3] - ySr2) ** 2 + (Y3[3] - ySr3) ** 2) / 3
    d5 = ((Y1[4] - ySr1) ** 2 + (Y2[4] - ySr2) ** 2 + (Y3[4] - ySr3) ** 2) / 3
    d6 = ((Y1[5] - ySr1) ** 2 + (Y2[5] - ySr2) ** 2 + (Y3[5] - ySr3) ** 2) / 3
    d7 = ((Y1[6] - ySr1) ** 2 + (Y2[6] - ySr2) ** 2 + (Y3[6] - ySr3) ** 2) / 3
    d8 = ((Y1[7] - ySr1) ** 2 + (Y2[7] - ySr2) ** 2 + (Y3[7] - ySr3) ** 2) / 3
    print("d1=", round(d1, 2), "d2=", round(d2, 2), "d3=", round(d3, 2), "d4=", round(d4, 2), "d5=", round(d5, 2),
          "d6=", round(d6, 2), "d7=", round(d7, 2), "d8=", round(d8, 2))

    dcouple = [d1, d2, d3, d4, d5, d6, d7, d8]

    m = 3
    Gp = max(dcouple) / sum(dcouple)
    f1 = m - 1
    f2 = N = 8
    Gt = 0.5157
    if Gp < Gt:
        print("Дисперсія однорідна")
    else:
        print("Дисперсія  неоднорідна")

    print("\nЗначимість коефіцієнтів регресії згідно критерію Стьюдента:")
    sb = sum(dcouple) / N
    sb2 = sb / N * m
    sbs = math.sqrt(sb2)

    beta0 = (ySr1 * 1 + ySr2 * 1 + ySr3 * 1 + ySr4 * 1 + ySr5 * 1 + ySr6 * 1 + ySr7 * 1 + ySr8 * 1) / 8
    beta1 = (ySr1 * (-1) + ySr2 * (-1) + ySr3 * (-1) + ySr4 * (-1) + ySr5 * 1 + ySr6 * 1 + ySr7 * 1 + ySr8 * 1) / 8
    beta2 = (ySr1 * (-1) + ySr2 * (-1) + ySr3 * 1 + ySr4 * 1 + ySr5 * (-1) + ySr6 * (-1) + ySr7 * 1 + ySr8 * 1) / 8
    beta3 = (ySr1 * (-1) + ySr2 * 1 + ySr3 * (-1) + ySr4 * 1 + ySr5 * (-1) + ySr6 * 1 + ySr7 * (-1) + ySr8 * 1) / 8
    beta4 = (ySr1 * 1 + ySr2 * 1 + ySr3 * (-1) + ySr4 * (-1) + ySr5 * (-1) + ySr6 * (-1) + ySr7 * 1 + ySr8 * 1) / 8
    beta5 = (ySr1 * 1 + ySr2 * (-1) + ySr3 * 1 + ySr4 * (-1) + ySr5 * (-1) + ySr6 * 1 + ySr7 * (-1) + ySr8 * 1) / 8
    beta6 = (ySr1 * 1 + ySr2 * (-1) + ySr3 * (-1) + ySr4 * 1 + ySr5 * 1 + ySr6 * (-1) + ySr7 * (-1) + ySr8 * 1) / 8
    beta7 = (ySr1 * (-1) + ySr2 * 1 + ySr3 * 1 + ySr4 * (-1) + ySr5 * 1 + ySr6 * (-1) + ySr7 * (-1) + ySr8 * 1) / 8

    t0 = abs(beta0) / sbs
    t1 = abs(beta1) / sbs
    t2 = abs(beta2) / sbs
    t3 = abs(beta3) / sbs
    t4 = abs(beta4) / sbs
    t5 = abs(beta5) / sbs
    t6 = abs(beta6) / sbs
    t7 = abs(beta7) / sbs

    # print(t0,t1,t2,t3)

    f3 = f1 * f2
    ttabl = 2.12

    d = 8
    if (t0 < ttabl):
        print("t0<ttabl, b0 не значимий")
        b0 = 0
        d = d - 1
    if (t1 < ttabl):
        print("t1<ttabl, b1 не значимий")
        b1 = 0
        d = d - 1
    if (t2 < ttabl):
        print("t2<ttabl, b2 не значимий")
        b2 = 0
        d = d - 1
    if (t3 < ttabl):
        print("t3<ttabl, b3 не значимий")
        b3 = 0
        d = d - 1
    if (t4 < ttabl):
        print("t4<ttabl, b4 не значимий")
        b4 = 0
        d = d - 1
    if (t5 < ttabl):
        print("t5<ttabl, b5 не значимий")
        b5 = 0
        d = d - 1
    if (t6 < ttabl):
        print("t6<ttabl, b6 не значимий")
        b6 = 0
        d = d - 1
    if (t7 < ttabl):
        print("t7<ttabl, b7 не значимий")
        b7 = 0
        d = d - 1
    yy1 = b0 + b1 * x1min + b2 * x2min + b3 * x3min + b4 * x1min * x2min + b5 * x1min * x3min + b6 * x2min * x3min + b7 * x1min * x2min * x3min
    yy2 = b0 + b1 * x1min + b2 * x2min + b3 * x3max + b4 * x1min * x2min + b5 * x1min * x3max + b6 * x2min * x3max + b7 * x1min * x2min * x3max
    yy3 = b0 + b1 * x1min + b2 * x2max + b3 * x3min + b4 * x1min * x2max + b5 * x1min * x3min + b6 * x2max * x3min + b7 * x1min * x2max * x3min
    yy4 = b0 + b1 * x1min + b2 * x2max + b3 * x3max + b4 * x1min * x2max + b5 * x1min * x3max + b6 * x2max * x3max + b7 * x1min * x2max * x3max
    yy5 = b0 + b1 * x1max + b2 * x2min + b3 * x3min + b4 * x1max * x2min + b5 * x1max * x3min + b6 * x2min * x3min + b7 * x1max * x2min * x3min
    yy6 = b0 + b1 * x1max + b2 * x2min + b3 * x3max + b4 * x1max * x2min + b5 * x1max * x3max + b6 * x2min * x3max + b7 * x1max * x2min * x3max
    yy7 = b0 + b1 * x1max + b2 * x2max + b3 * x3min + b4 * x1max * x2max + b5 * x1max * x3min + b6 * x2max * x3min + b7 * x1max * x2min * x3max
    yy8 = b0 + b1 * x1max + b2 * x2max + b3 * x3max + b4 * x1max * x2max + b5 * x1max * x3max + b6 * x2max * x3max + b7 * x1max * x2max * x3max
    print("\nКритерій Фішера")
    print(d, " значимих коефіцієнтів")
    f4 = N - d

    sad = ((yy1 - ySr1) ** 2 + (yy2 - ySr2) ** 2 + (yy3 - ySr3) ** 2 + (yy4 - ySr4) ** 2 + (yy5 - ySr5) ** 2 + (
                yy6 - ySr6) ** 2 + (yy7 - ySr7) ** 2 + (yy8 - ySr8) ** 2) * (m / (N - d))
    Fp = sad / sb
    print("Fp=", round(Fp, 2))
    F = [4.5, 3.6, 3.2, 3.0, 2.9, 2.7, 2.4, 2.2]
    Fi = [1, 2, 3, 4, 5, 6, 12, 24]
    dif = 100
    counter = 0
    for i in range(len(F)):
        if abs(Fi[i] - (i + 1)) < dif:
            dif = abs(Fi[i] - i)
            counter = i
    Ft = F[counter]
    cont = 0
    if Fp > Ft:
        print("Fp=", round(Fp, 2), ">Ft", Ft, "Рівняння неадекватно оригіналу")
        cont = 1
    else:
        print("Fp=", round(Fp, 2), "<Ft", Ft, "Рівняння адекватно оригіналу")
