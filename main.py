import random
import statistics as st
import itertools

import numpy as np
import scipy
from scipy.stats import f
from scipy.stats import t
from prettytable import PrettyTable


def f_critical(prob, f1, f2):
    return scipy.stats.f.ppf(prob, f1, f2)


def t_critical(prob, df):
    return scipy.stats.t.ppf(prob, df)


def c_critical(prob, f1, f2):
    return 1 / (
        1 + (f2 - 1) / scipy.stats.f.ppf(1 - (1 - prob) / f2, f1, (f2 - 1) * f1)
    )


x_bounds = [[1, 1], [-30, 0], [-25, 10], [-25, -5]]
factors = len(x_bounds)
N = 4
m = 2
confidence_prob = 0.9

combinations = list(itertools.product([-1, 1], repeat=4))
xn = [combinations[8], combinations[11], combinations[13], combinations[14]]
x = [
    [
        min(x_bounds[j]) if xn[i][j] < 0 else max(x_bounds[j])
        for j in range(len(x_bounds))
    ]
    for i in range(len(xn))
]

y_bounds = [
    int(200 + st.mean([min(x_bounds[i]) for i in range(1, factors)])),
    int(200 + st.mean([max(x_bounds[i]) for i in range(1, factors)])),
]


def create_matrix():
    table = PrettyTable()
    table_head = ["Експеремент #"]
    for i in range(factors):
        table_head.append(f"x{i}")

    for i in range(m):
        table_head.append(f"y{i+1}")

    table.field_names = table_head

    for i in range(N):
        table.add_row([i + 1, *xn[i], *y[i]])

    return table


###
### Критерій Кохрена
###
y = [
    [random.randint(min(y_bounds), max(y_bounds)) for i in range(m)]
    for j in range(N)
]
while True:

    matrix = create_matrix()
    print(matrix)

    s2_y = [st.variance(y[i]) for i in range(N)]
    stat_c = max(s2_y) / sum(s2_y)
    crit_c = c_critical(confidence_prob, m - 1, N)
    print(f"Критерій Кохрена: {round(stat_c, 3)}")
    print(
        f"Критичне значення критерію Кохрена (для {confidence_prob}): {round(crit_c, m)}"
    )

    if stat_c < crit_c:
        print("Дисперсія однорідна.")
        break
    if m > 23: exit()

    print("Дисперсія не однорідна. Збільшуємо m")
    m += 1 # if Gp(stat_c) > Gt(crit_c) m+1
    for yi in y:
        yi.append(random.randint(min(y_bounds), max(y_bounds)))

my = [sum(y[i]) / len(y[i]) for i in range(len(y))]
xn_col = np.array(list(zip(*xn)))

beta = [sum(my * xn_col[i]) / len(my * xn_col[i]) for i in range(N)]
yn = [sum(beta * np.array(xn[i])) for i in range(N)]

delta_x = [abs(x_bounds[i][0] - x_bounds[i][1]) / 2 for i in range(len(x_bounds))]
x0 = [(x_bounds[i][0] + x_bounds[i][1]) / 2 for i in range(len(x_bounds))]

# натуралізована регресія
b = [beta[0] - sum(beta[i] * x0[i] / delta_x[i] for i in range(1, factors))]
b.extend([beta[i] / delta_x[i] for i in range(1, factors)])

###
### Критерій Стьюдента
###

s2_b = sum(s2_y) / len(s2_y)
s_beta = np.sqrt(s2_b / m / N)
stat_t = [abs(beta[i]) / s_beta for i in range(factors)]
crit_t = t_critical(confidence_prob, (m - 1) * N)
significant_coeffs = 4

print(f"Критерій Стьюдента: {[round(stat_t[i], 3) for i in range(len(stat_t))]}")
print(f"Критичне значення критерія Стьюдента (для {confidence_prob}): {round(crit_t, 3)}")

for i in range(len(stat_t)):
    if stat_t[i] < crit_t:
        b[i] = 0
        significant_coeffs -= 1

print(f"Коефіцієнти регресії: {[round(b[i], 3) for i in range(len(b))]}")

y_calc = [sum((b * np.array(x))[i]) for i in range(N)]
print(
    f"Розрахункове значення функції: {[round(y_calc[i], 3) for i in range(len(y_calc))]}"
)

s2_adeq = (
    m
    / (N - significant_coeffs)
    * sum([(y_calc[i] - my[i]) ** 2 for i in range(N)])
)

stat_f = s2_adeq / s2_b
crit_f = f_critical(
    confidence_prob, (m - 1) * N, N - significant_coeffs
)

print(f"Критерій Фішера: {round(stat_f, 3)}")
print(f"Критичне значення критерія Фішера (для {confidence_prob}): {round(crit_f, 3)}")

if stat_f > crit_f:
    print("Модель не адекватна.")
else:
    print("Модель адакватна.")
