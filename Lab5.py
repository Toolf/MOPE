import random
from prettytable import PrettyTable
from scipy.stats import f, t
from functools import partial

x1min, x1max = -4, 6
x2min, x2max = -1, 2
x3min, x3max = -4, 2

x_aver_min = (x1min + x2min + x3min) / 3
x_aver_max = (x1max + x2max + x3max) / 3

x1_aver = (x1max + x1min) / 2
x2_aver = (x2max + x2min) / 2
x3_aver = (x3max + x3min) / 2

x1del = x1max - x1_aver
x2del = x2max - x2_aver
x3del = x3max - x3_aver

y_min = 200 + int(x_aver_min)
y_max = 200 + int(x_aver_max)

m = 3
while m <20:
    print("Рівняння регресії")
    print("y=b0+b1*x1+b2*x2+b3*x3+b12*x1*x2+b13*x1*x3+b23*x2*x3+b123*x1*x2*x3+b11*x1^2+b22*x2^2+b33*x3^2")

    Y_ALL = [[random.randint(y_min, y_max) for _ in range(15)] for _ in range(m)]

    x0_norm = [1] * 15
    x1_norm = [-1, -1, -1, -1, 1, 1, 1, 1, -1.215, 1.215, 0, 0, 0, 0, 0]
    x2_norm = [-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, -1.215, 1.215, 0, 0, 0]
    x3_norm = [-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -1.215, 1.215, 0]
    x12_norm = [x * y for x, y in zip(x1_norm, x2_norm)]
    x13_norm = [x * y for x, y in zip(x1_norm, x3_norm)]
    x23_norm = [x * y for x, y in zip(x2_norm, x3_norm)]
    x123_norm = [x * y * z for x, y, z in zip(x1_norm, x2_norm, x3_norm)]
    x11_norm = [x * x for x in x1_norm]
    x22_norm = [x * x for x in x2_norm]
    x33_norm = [x * x for x in x3_norm]

    def x_filler (arr, xmin, xmax, xdel, x_aver):
        l = []
        for i in arr:
            if i == -1:
                l.append(xmin)
            elif abs(i) == 1.215:
                l.append(i * xdel + x_aver)
            else:
                l.append(xmax)
        return l

    x1 = x_filler(x1_norm, x1min, x1max, x1del, x1_aver)
    x2 = x_filler(x2_norm, x2min, x2max, x2del, x2_aver)
    x3 = x_filler(x3_norm, x3min, x3max, x3del, x3_aver)
    x12 = [x * y for x, y in zip(x1, x2)]
    x13 = [x * y for x, y in zip(x1, x3)]
    x23 = [x * y for x, y in zip(x2, x3)]
    x123 = [x * y * z for x, y, z in zip(x1, x2, x3)]
    x11 = [x * x for x in x1]
    x22 = [x * x for x in x2]
    x33 = [x * x for x in x3]

    def table (names, values):
        pretty = PrettyTable()
        for i in range(len(names)):
            pretty.add_column(names[i], values[i])
        print(pretty, "\n")

    names = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "X1^2", "X2^2", "X3^2"]
    values = [x0_norm, x1_norm, x2_norm, x3_norm, x12_norm, x13_norm, x23_norm, x123_norm, x11_norm, x22_norm, x33_norm]
    table(names, values)

    print(f"Матриця для m={m}")

    def re_zip (allYValues):
        l = [[0 for _ in range(len(allYValues))] for _ in range(len(allYValues[0]))]
        for i in range(len(allYValues)):
            for j in range(len(allYValues[i])):
                l[j][i] = allYValues[i][j]
        return l

    Y_ALL = [[random.randint(y_min, y_max) for _ in range(15)] for _ in range(m)]
    Y_ROWS = re_zip(Y_ALL)
    Y_ROWS_AV = [sum(x) / len(x) for x in Y_ROWS]

    for i in range(len(Y_ALL)):
        names.append(f"Y{i+1}")
        values.append(Y_ALL[i])
    names.append("Y_AVERAGE")
    values.append(Y_ROWS_AV)

    table(names, values)
    #################################################################################
    disp = [0] * 15

    for i in range(15):
        disp[i] = sum([(Y_ROWS_AV[i] - Y_ROWS[i][j]) ** 2 for j in range(m)]) / m

    Gp = max(disp) / sum(disp)

    f1 = m - 1
    f2 = N = 15

    def cohren_teoretical (f1, f2, q=0.05):
        q1 = q / f1
        fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
        return fisher_value / (fisher_value + f1 - 1)

    Gt = cohren_teoretical(f1, f2)
    print("Дисперсія по рядкам")
    for i, j in enumerate(disp):
        print(f"{i+1}. {j:.2f}")
    if Gp < Gt:
        print("Дисперсія однорідна")
    else:
        print("Дисперсія неоднорідна")

    print("Критерій Стьюдента")
    sb = sum(disp) / N
    ssbs = sb / (m * N)
    sbs = ssbs ** 0.5

    bethas = [0] * 11
    x_norm = [x0_norm, x1_norm, x2_norm, x3_norm, x12_norm, x13_norm, x23_norm, x123_norm, x11_norm, x22_norm, x33_norm]
    for i in range(11):
        for j in range(len(x1_norm)):
            bethas[i] += Y_ROWS_AV[j]*x_norm[i][j]
        bethas[i] /= 15

    tethas = [abs(bethas[i]) / sbs for i in range(len(bethas))]

    f3 = f1 * f2
    student_teoretical = partial(t.ppf, q=1 - 0.025)
    T = student_teoretical(df=f3)
    d = 0

    for i in range(len(tethas)):
        if tethas[i] < T:
            bethas[i] = 0
            print(f"Приймаємо betha{i} незначимим")
        else:
            print(f"Betha{i} = {bethas[i]}")
            d += 1

    yy1 = bethas[0] + bethas[1] * x1min + bethas[2] * x2min + bethas[3] * x3min + bethas[4] * x1min * x2min + bethas[
        5] * x1min * x3min + bethas[
              6] * x2min * x3min + bethas[7] * x1min * x2min * x3min + bethas[8] * x1min * x1min + bethas[
              9] * x2min * x2min + bethas[
              10] * x3min * x3min
    yy2 = bethas[0] + bethas[1] * x1min + bethas[2] * x2min + bethas[3] * x3max + bethas[4] * x1min * x2min + bethas[
        5] * x1min * x3max + bethas[
              6] * x2min * x3max + bethas[7] * x1min * x2min * x3max + bethas[8] * x1min * x1min + bethas[
              9] * x2min * x2min + bethas[
              10] * x3max * x3max
    yy3 = bethas[0] + bethas[1] * x1min + bethas[2] * x2max + bethas[3] * x3min + bethas[4] * x1min * x2max + bethas[
        5] * x1min * x3min + bethas[
              6] * x2max * x3min + bethas[7] * x1min * x2max * x3min + bethas[8] * x1min * x1min + bethas[
              9] * x2max * x2max + bethas[
              10] * x3min * x3min
    yy4 = bethas[0] + bethas[1] * x1min + bethas[2] * x2max + bethas[3] * x3max + bethas[4] * x1min * x2max + bethas[
        5] * x1min * x3max + bethas[
              6] * x2max * x3max + bethas[7] * x1min * x2max * x3max + bethas[8] * x1min * x1min + bethas[
              9] * x2max * x2max + bethas[
              10] * x3max * x3max
    yy5 = bethas[0] + bethas[1] * x1max + bethas[2] * x2min + bethas[3] * x3min + bethas[4] * x1max * x2min + bethas[
        5] * x1max * x3min + bethas[
              6] * x2min * x3min + bethas[7] * x1max * x2min * x3min + bethas[8] * x1max * x1max + bethas[
              9] * x2min * x2min + bethas[
              10] * x3min * x3min
    yy6 = bethas[0] + bethas[1] * x1max + bethas[2] * x2min + bethas[3] * x3max + bethas[4] * x1max * x2min + bethas[
        5] * x1max * x3max + bethas[
              6] * x2min * x3max + bethas[7] * x1max * x2min * x3max + bethas[8] * x1max * x1max + bethas[
              9] * x2min * x2min + bethas[
              10] * x3min * x3max
    yy7 = bethas[0] + bethas[1] * x1max + bethas[2] * x2max + bethas[3] * x3min + bethas[4] * x1max * x2max + bethas[
        5] * x1max * x3min + bethas[
              6] * x2max * x3min + bethas[7] * x1max * x2min * x3max + bethas[8] * x1max * x1max + bethas[
              9] * x2max * x2max + bethas[
              10] * x3min * x3min
    yy8 = bethas[0] + bethas[1] * x1max + bethas[2] * x2max + bethas[3] * x3max + bethas[4] * x1max * x2max + bethas[
        5] * x1max * x3max + bethas[
              6] * x2max * x3max + bethas[7] * x1max * x2max * x3max + bethas[8] * x1max * x1max + bethas[
              9] * x2max * x2max + bethas[
              10] * x3min * x3max
    yy9 = bethas[0] + bethas[1] * x1[8] + bethas[2] * x2[8] + bethas[3] * x3[8] + bethas[4] * x12[8] + bethas[5] * x13[
        8] + bethas[6] * x23[8] + bethas[7] * \
          x123[8] + bethas[8] * x11[8] + bethas[9] * x22[8] + bethas[10] * x33[8]
    yy10 = bethas[0] + bethas[1] * x1[9] + bethas[2] * x2[9] + bethas[3] * x3[9] + bethas[4] * x12[9] + bethas[5] * x13[
        9] + bethas[6] * x23[9] + bethas[7] * \
           x123[9] + bethas[8] * x11[9] + bethas[9] * x22[9] + bethas[10] * x33[9]
    yy11 = bethas[0] + bethas[1] * x1[10] + bethas[2] * x2[10] + bethas[3] * x3[10] + bethas[4] * x12[10] + bethas[5] * \
           x13[10] + bethas[6] * x23[10] + bethas[
               7] * x123[10] + bethas[8] * x11[10] + bethas[9] * x22[10] + bethas[10] * x33[10]
    yy12 = bethas[0] + bethas[1] * x1[11] + bethas[2] * x2[11] + bethas[3] * x3[11] + bethas[4] * x12[11] + bethas[5] * \
           x13[11] + bethas[6] * x23[11] + bethas[
               7] * x123[11] + bethas[8] * x11[11] + bethas[9] * x22[11] + bethas[10] * x33[11]
    yy13 = bethas[0] + bethas[1] * x1[12] + bethas[2] * x2[12] + bethas[3] * x3[12] + bethas[4] * x12[12] + bethas[5] * \
           x13[12] + bethas[6] * x23[12] + bethas[
               7] * x123[12] + bethas[8] * x11[12] + bethas[9] * x22[12] + bethas[10] * x33[12]
    yy14 = bethas[0] + bethas[1] * x1[13] + bethas[2] * x2[13] + bethas[3] * x3[13] + bethas[4] * x12[13] + bethas[5] * \
           x13[13] + bethas[6] * x23[13] + bethas[
               7] * x123[13] + bethas[8] * x11[13] + bethas[9] * x22[13] + bethas[10] * x33[13]
    yy15 = bethas[0] + bethas[1] * x1[14] + bethas[2] * x2[14] + bethas[3] * x3[14] + bethas[4] * x12[14] + bethas[5] * \
           x13[14] + bethas[6] * x23[14] + bethas[
               7] * x123[14] + bethas[8] * x11[14] + bethas[9] * x22[14] + bethas[10] * x33[14]

    print("Критерій Фішера")
    f4 = N - d
    yy = [yy1, yy2, yy3, yy4, yy5, yy6, yy7, yy8, yy9, yy10, yy11, yy12, yy13, yy14, yy15]
    sad = sum([(yy[i] - Y_ROWS_AV[i]) ** 2 for i in range(len(yy))]) * m / (N - d)
    Fp = sad / sb
    fisher_teoretical = partial(f.ppf, q=1 - 0.05)
    Ft = fisher_teoretical(dfn=f4, dfd=f3)
    if Ft > Fp:
        print("Рівняння регресії адекватне оригіналу")
        break
    else:
        print("Рівняння регресії не є адекватне оригіналу")
        m += 1
