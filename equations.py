"""
Sympy code used to derive some expressions used in the implementation.
"""

from sympy import symbols, Eq, simplify, expand, solve, collect, bottom_up, Number, S, poly, nonlinsolve, plot, sqrt
from pprint import pprint

def dist(p, x, y):
    px, py = p
    return sqrt((x - px)**2 + (y - py)**2)

# get the line equation of the bisector of two points
def bisector(p1, p2, x, y):
    return Eq(dist(p1, x, y) - dist(p2, x, y), 0)

def star_map(equation, p, x, y):
    y_of_x = solve(equation, y)[0]
    print('y_of_x:', y_of_x)
    return Eq(y, y_of_x + dist((0,0), x, y_of_x))


def s(line, x, y):
    c_ = lambda l: bottom_up(l, lambda e: collect(e, [x,y]))
    s_ = lambda l: c_(expand(l))
    if isinstance(line, Eq):
        return Eq(s_(line.lhs), s_(line.rhs))
    return s_(line)

# From https://stackoverflow.com/a/63480143
def collect_with_respect_to_vars(eq, vars):
    assert isinstance(vars, list)
    eq = eq.expand()
    if len(vars) == 0:
        return {1: eq}

    var_map = eq.collect(vars[0], evaluate=False)
    final_var_map = {}
    for var_power in var_map:
        sub_expression = var_map[var_power]
        sub_var_map = collect_with_respect_to_vars(sub_expression, vars[1:])
        for sub_var_power in sub_var_map:
            final_var_map[var_power*sub_var_power] = sub_var_map[sub_var_power]
    return final_var_map

def bisectors_intersection():
    x, y = symbols('x y', real=True)
    px, py, qx, qy, rx, ry, sx, sy = symbols('px py qx qy rx ry sx sy', real=True)

    bpq = bisector((px, py), (qx, qy), x, y)
    brs = bisector((rx, ry), (sx, sy), x, y)

    print(bpq)
    print(brs)
    
    intersection = solve([bpq, brs], [x, y])

    print('intersection', intersection)

def bisector_star_compare():
    """
    Given the star-map of a bisector, check if a point is above or below the bisector
    """

    x, y = symbols('x y', real=True)
    px, py, qx, qy = symbols('px py qx qy', real=True)
    sx, sy = symbols('sx sy', real=True)

    bpq = bisector((px, py), (0, 0), x, y)

    # star-map of the bisector
    star_map_bpq = star_map(bpq, (0, 0), x, y)

    print('star_map_bpq:', star_map_bpq)

    star_by = solve(star_map_bpq.subs({x: sx}), y)[0]

    print('by:', star_by)

    # point to check
    cmp = star_by - sy

    # check if the point is above or below the bisector
    print('cmp:', cmp)

# bisectors_intersection()
bisector_star_compare()
