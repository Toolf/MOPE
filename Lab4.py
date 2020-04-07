import random
from prettytable import PrettyTable
import numpy
from scipy.stats import f, t
import math
from functools import partial

var = True
m = 3

while var:

    x1min = -5
    x1max = 15
    x2min = 10
    x2max = 60
    x3min = 10
    x3max = 20
    xAvmin = (x1min + x2min + x3min) / 3  # Xcpmin
    xAvmax = (x1max + x2max + x3max) / 3  # Xcpmax

    ymin = int(200 + xAvmin)
    ymax = int(200 + xAvmax)

    print("Рівняння регресії з ефектом взаємодії")
    print("y=b0+b1*x1+b2*x2+b3*x3+b12*x1*x2+b13*x1*x3+b23*x2*x3+b123*x1*x2*x3+b11*x1^2+b22*x2^2+b33*x3^2")

    # Матриця ПФЕ
    x0_cod = [1, 1, 1, 1, 1, 1, 1, 1]
    x1_cod = [-1, -1, 1, 1, -1, -1, 1, 1]
    x2_cod = [-1, 1, -1, 1, -1, 1, -1, 1]
    x3_cod = [-1, 1, 1, -1, 1, -1, -1, 1]
    x1x2_cod = [x * y for x, y in zip(x1_cod, x2_cod)]
    x1x3_cod = [x * y for x, y in zip(x1_cod, x3_cod)]
    x2x3_cod = [x * y for x, y in zip(x2_cod, x3_cod)]
    x1x2x3_cod = [x * y * z for x, y, z in zip(x1_cod, x2_cod, x3_cod)]

    Y_ALL = [[] for _ in range(m)]
    for j in range(m):
        for i in range(8):
            Y_ALL[j].append(random.randint(ymin, ymax))


    def re_zip(allYValues):
        l = [[0 for _ in range(len(allYValues))] for _ in range(len(allYValues[0]))]
        for i in range(len(allYValues)):
            for j in range(len(allYValues[i])):
                l[j][i] = allYValues[i][j]
        return l


    Y_ROWS = re_zip(Y_ALL)
    Y_ROWS_AV = [sum(x) / len(x) for x in Y_ROWS]
    x0 = [1, 1, 1, 1, 1, 1, 1, 1]
    x1 = [x1min if i == -1 else x1max for i in x1_cod]
    x2 = [x2min if i == -1 else x2max for i in x2_cod]
    x3 = [x3min if i == -1 else x3max for i in x3_cod]
    x1x2 = [x * y for x, y in zip(x1, x2)]
    x1x3 = [x * y for x, y in zip(x1, x3)]
    x2x3 = [x * y for x, y in zip(x2, x3)]
    x1x2x3 = [x * y * z for x, y, z in zip(x1, x2, x3)]

    list_cod = [x0_cod, x1_cod, x2_cod, x3_cod, x1x2_cod, x1x3_cod, x2x3_cod, x1x2x3_cod]

    N = 8
    bethas = []
    print(Y_ROWS_AV)
    for i in range(N):
        S = 0
        for j in range(N):
            S += (list_cod[i][j] * Y_ROWS_AV[j]) / N
        bethas.append(S)

    DISPS = []
    for j in range(8):
        DISPS.append(sum([(Y_ROWS[j][i] - Y_ROWS_AV[j]) ** 2 for i in range(m)])/m)

    sum_dispersion = sum(DISPS)


    def table(names, values):
        pretty = PrettyTable()
        for i in range(len(names)):
            pretty.add_column(names[i], values[i])
        print(pretty, "\n")


    column_names = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "Y AVERAGE"]
    for i in range(len(Y_ALL)):
        column_names.append(f"Y{i + 1}")
    column_names.append("S^2")
    ALL1 = [x0_cod, x1_cod, x2_cod, x3_cod, x1x2_cod, x1x3_cod, x2x3_cod, x1x2x3_cod, Y_ROWS_AV]
    ALL2 = [x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3, Y_ROWS_AV]
    for i in Y_ALL:
        ALL2.append(i)
        ALL1.append(i)
    ALL2.append(DISPS)
    ALL1.append(DISPS)
    table(column_names, ALL1)

    # рівняння регресії з ефектом взаємодії
    print("y = {:.3f} + {:.3f}*x1 + {:.3f}*x2 "
          "+ {:.3f}*x3 + {:.3f}*x1x2 + {:.3f}*x1x3 "
          "+ {:.3f}*x2x3 + {:.3f}*x1x2x3 \n".format(bethas[0], bethas[1],
                                                    bethas[2], bethas[3],
                                                    bethas[4], bethas[5],
                                                    bethas[6], bethas[7]))

    table(column_names, ALL2)
    left = list(zip(x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3))
    alphas = [i for i in numpy.linalg.solve(left, Y_ROWS_AV)]
    print("y = {:.3f} + {:.3f}*x1 + {:.3f}*x2 "
          "+ {:.3f}*x3 + {:.3f}*x1x2 + {:.3f}*x1x3 "
          "+ {:.3f}*x2x3 + {:.3f}*x1x2x3 \n".format(alphas[0], alphas[1],
                                                    alphas[2], alphas[3],
                                                    alphas[4], alphas[5],
                                                    alphas[6], alphas[7]))

    Gp = max(DISPS) / sum(DISPS)
    F1 = m - 1
    N = len(Y_ALL[0])
    F2 = N


    def cohren_teoretical(f1, f2, q=0.05):
        q1 = q / f1
        fisher_value = f.ppf(q=1 - q1, dfn=f2, dfd=(f1 - 1) * f2)
        return fisher_value / (fisher_value + f1 - 1)


    Gt = cohren_teoretical(F1, F2)

    print("\nGp = ", Gp, " Gt = ", Gt)
    if Gp < Gt:
        print("Дисперсія однорідна")

        Sb = sum(DISPS) / N
        SSb = Sb / (m * N)
        notSSb = math.sqrt(SSb)

        betas = [0 for _ in range(8)]
        for i in range(len(x0_cod)):
            betas[0] += Y_ROWS_AV[i] * x0_cod[i] / N
            betas[1] += Y_ROWS_AV[i] * x1_cod[i] / N
            betas[2] += Y_ROWS_AV[i] * x2_cod[i] / N
            betas[3] += Y_ROWS_AV[i] * x3_cod[i] / N
            betas[4] += Y_ROWS_AV[i] * x1x2_cod[i] / N
            betas[5] += Y_ROWS_AV[i] * x1x3_cod[i] / N
            betas[6] += Y_ROWS_AV[i] * x2x3_cod[i] / N
            betas[7] += Y_ROWS_AV[i] * x1x2x3_cod[i] / N

        tethas = [abs(i) / notSSb for i in betas]

        F3 = F1 * F2
        d = 0

        student_teoretical = partial(t.ppf, q=1 - 0.025)
        T = student_teoretical(df=F3)
        for i in range(len(tethas)):
            if tethas[i] < T:
                betas[i] = 0
                print(f"Приймаємо betha{i} незначимим")
            else:
                print(f"Betha{i} = {betas[i]}")
                d += 1

        coef = [[1 for _ in range(len(x1))], x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3]
        y_for_Student = [0 for _ in range(8)]
        for i in range(len(betas)):
            for j in range(len(coef[0])):
                y_for_Student[i] += betas[j] * coef[j][i]
        F4 = N - d
        Disp_ad = 0
        for i in range(len(y_for_Student)):
            Disp_ad += ((y_for_Student[i] - Y_ROWS_AV[i]) ** 2) * m / (N - d)
        Fp = Disp_ad / SSb
        fisher_teoretical = partial(f.ppf, q=1 - 0.05)
        Ft = fisher_teoretical(dfn=F4, dfd=F3)
        if Ft > Fp:
            print("Рівняння регресії адекватне оригіналу")
        else:
            print("Рівняння регресії не є адекватне оригіналу")
        var = False

    else:
        print("Дисперсія не однонідна. Збільшити m")
        m += 1
