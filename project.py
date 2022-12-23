'''Project discretka'''
import math
import numpy as np
from numpy.linalg import matrix_power
import timeit 


def linear_equationa(coefs):
    '''
    Finds roots of linear equation.
    '''
    return [-(coefs[0] / coefs[1])]

def quadratic_equation(coefs:  list[int]):
    '''
    Finds roots of quadratic equation.
    '''
    try:
        discr = coefs[1]**2 - 4*coefs[0]*coefs[2]
        x1 = (-coefs[1] - math.sqrt(discr)) / 2*coefs[2]
        x2 = (-coefs[1] + math.sqrt(discr)) / 2*coefs[2]
    except ValueError:
        return 'No real roots.'
    return [x1, x2]


def roots_of_polynomial(coefs):
    '''
    Finds integer roots of polynomial equation.
    '''
    roots = []
    for _i in range(-1000, 1001):
        result = 0
        for deg, coef in enumerate(coefs):
            result += coef*(_i**deg)
        if not result:
            roots.append(_i)
    return roots


def find_c(coefs, answers):
    '''
    Function info.
    '''
    deg = len(answers)
    equations = []
    for _i1 in range(deg):
        equations.append([coef**_i1 for coef in coefs] + [answers[_i1]])
    general_matrix = np.array([matrix[:deg] for matrix in equations])
    det_gm = round(np.linalg.det(general_matrix))
    dms = []
    for _i2 in range(deg):
        matrix = list(zip(*general_matrix))
        matrix[_i2] = tuple(answers)
        dms.append(round(np.linalg.det(list(zip(*matrix)))))
    results = [round(val/det_gm,3) for val in dms]
    return results


def result(lst: list[int], answers: list[int], n: int) -> list[int]:
    """
    Finds n first roots of recursive equation
    >>> result([-3,2,1], [1,2], 8)
    [1, 2, -1.0, 8.0, -19.0, 62.0, -181.0, 548.0]
    """
    roots = quadratic_equation(lst)
    koef = find_c(roots, answers)
    result=answers
    for i in range(len(answers),n):
        total = 0
        for number,element in enumerate(roots):
            total+=(element**i)*koef[number]
        result.append(total)
    return result


def matrix(lst: list) -> list[list]:
    """
    Creates transitive matrix
    for coefficients of recursive equation
    >>> matrix([-3, 2, 1])
    [[0, 3.0], [1, -2.0]]
    """
    lst[-1] = -lst[-1]
    res = []
    tipchik = -1
    for i in range(len(lst)-1):
        add_res = []
        for j in range(len(lst)-1):
            if tipchik != j:
                if j != len(lst)-2:
                    add_res += [0]
                else:
                    add_res += [lst[i]/lst[-1]]
            else:
                add_res += [1]
        res += [add_res]
        tipchik += 1
    return res

def get_nth_el(first_two_el: list, transist_matrix: list, n: int) -> float:
    """
    Finds nth element of recursive equation
    >>> get_nth_el([1,2], matrix[-3,2,1], 8)
    548.0
    """ 
    matrix = matrix_power(np.array(transist_matrix), n-2) 
    n_th = list(np.matmul(np.array(first_two_el), matrix))[-1] 
    return n_th 
 
print(f'O(logn) difficulty time= {timeit.timeit(str(get_nth_el([1, 2], matrix([-3,2,1]), 8)))}')
print(f'Analytic method time = {timeit.timeit(str(result([-3,2,1], [1,2], 8)))}')
