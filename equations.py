"""
Sympy code used to derive some expressions used in the implementation.
"""

from sympy import symbols, Eq, simplify, expand, solve, collect, bottom_up, Number, S, poly, nonlinsolve, plot, sqrt, cse
import sympy as sp
from pprint import pprint

def cse_to_c_code(expression):
    # Perform CSE on the given expression
    replacements, reduced_expr = sp.cse(list(expression))

    print('replacements:', replacements)
    print('reduced_expr:', reduced_expr)
    
    # Generate C code for the replacements (subexpressions)
    c_code_lines = []
    for var, subexpr in replacements:
        c_code_lines.append(f"double {var} = {sp.ccode(subexpr)};")
    
    # Generate C code for the final expression
    c_code_lines.append(f"double x = {sp.ccode(reduced_expr[0])};")
    c_code_lines.append(f"double y = {sp.ccode(reduced_expr[1])};")
    
    # Combine all lines into the final C code string
    c_code = "\n".join(c_code_lines)
    return c_code

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

    ccode = cse_to_c_code(intersection[0])

    print('ccode:', ccode)

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

def bisector_star_at_y():
    """
    Give the star-map of a bisector, find the x-coordinate of the bisector at a given y-coordinate
    """
    x, y = symbols('x y', real=True)
    dx, dy = symbols('dx dy', real=True)
    py = symbols('py', real=True)

    bpq = bisector((dx, dy), (0, 0), x, y)

    # star-map of the bisector
    star_map_bpq = star_map(bpq, (0, 0), x, y)

    print('star_map_bpq:', star_map_bpq)

    star_bx = solve(star_map_bpq.subs({y: py}), x)

    print('bx:', star_bx[0])
    print('bx:', star_bx[1])


# bisectors_intersection()
# bisector_star_compare()
bisector_star_at_y()
