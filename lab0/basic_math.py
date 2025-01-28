from scipy.optimize import minimize_scalar


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError('Умножение выполнить невозможно.')

    matrix_c = []
    for i in range(len(matrix_a)):
        row = []
        for k in range(len(matrix_b[0])):
            el = 0
            for j in range(len(matrix_b)):
                el += matrix_a[i][j] * matrix_b[j][k]
            row.append(el)
        matrix_c.append(row)

    return matrix_c


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    coeffs_1 = list(map(int, a_1.split()))
    coeffs_2 = list(map(int, a_2.split()))

    if coeffs_1 == coeffs_2:
        return None  # бесконечное множество решений

    a1, b1, c1 = coeffs_1
    a2, b2, c2 = coeffs_2

    def f(x, coeff):
        return coeff[0] * x ** 2 + coeff[1] * x + coeff[2]

    extremum_1 = minimize_scalar(f, args=coeffs_1, bounds=(-100, 100), method='bounded')
    extremum_2 = minimize_scalar(f, args=coeffs_2, bounds=(-100, 100), method='bounded')

    if extremum_1.success and extremum_2.success:
        if abs(extremum_1.x - (-100.0)) < 1e-5:
            print("Не удалось найти экстремумы.")
            print(f'Точка экстремума P(x): ({extremum_2.x:.2f}; {f(extremum_2.x, coeffs_2):.2f})')
        else:
            print(f'Точка экстремума F(x): ({extremum_1.x:.2f}; {f(extremum_1.x, coeffs_1):.2f})')
            if abs(extremum_2.x - (-100.0)) < 1e-5:
                print("Не удалось найти экстремумы.")
            else:
                print(f'Точка экстремума P(x): ({extremum_2.x:.2f}; {f(extremum_2.x, coeffs_2):.2f})')

    a = a1 - a2
    b = b1 - b2
    c = c1 - c2

    if a != 0:
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return []  # общих решений нет
        elif discriminant == 0:
            x = -b / (2 * a)
            return [(x, a1 * x ** 2 + b1 * x + c1)]
        else:
            x1 = (-b + discriminant ** 0.5) / (2 * a)
            x2 = (-b - discriminant ** 0.5) / (2 * a)
            return [(x1, a1 * x1 ** 2 + b1 * x1 + c1), (x2, a1 * x2 ** 2 + b1 * x2 + c1)]
    elif b != 0:
        x = -c / b
        return [(x, a1 * x ** 2 + b1 * x + c1)]
    else:
        # если a и b = 0, то функции не имеют общих решений
        return []


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    if n == 0:
        return None  # пустой списка

    x_mean = sum(x) / n

    m3 = sum((i - x_mean) ** 3 for i in x) / n
    m2 = sum((i - x_mean) ** 2 for i in x) / n

    if m2 == 0:
        return None  # стандартное отклонение равно нулю

    asymmetry_coefficient = m3 / (m2 ** 1.5)
    return round(asymmetry_coefficient, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    if n == 0:
        return None  # пустой список

    x_mean = sum(x) / n

    m4 = sum((i - x_mean) ** 4 for i in x) / n
    m2 = sum((i - x_mean) ** 2 for i in x) / n

    if m2 == 0:
        return None  # стандартное отклонение равно нулю

    excess_kurtosis = m4 / (m2 ** 2) - 3
    return round(excess_kurtosis, 2)
