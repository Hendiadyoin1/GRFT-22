import numpy as np
from numpy import pi, sqrt, cbrt
from matplotlib import pyplot as plt

µ_E = 398600.4418 * 1e3**3  # m³ / s²
r_E = 6371e3                # m
J2_E = 0.00108263

sidereal_day = 23.93447192 * 60 * 60 # s

µ_Moon = 4.9e12    # m³ / s²
r_Moon = 1_738_100 # m
h_Moon = 384_400_000 


def linear_eccentricity(a, b):
    return np.sqrt(a**2 - b**2)


def numerical_eccentricity(a, b):
    return linear_eccentricity(a, b) / a

def numerical_eccentricity_from_rarp(ra,rp):
    return 1 - (2 * rp/(ra+rp))

def semi_latus_rectum(a, b):
    return b**2 / a

def apoapsis(a, b):
    return linear_eccentricity(a, b) + a

def periapsis(a, b):
    return a - linear_eccentricity(a, b)

def orbital_speed_circle(a, *, µ=µ_E):
    return np.sqrt(µ / a)

def orbital_period_circle(a, *, µ=µ_E):
    return 2 * pi * a / orbital_speed_circle(a)

def orbital_height_circle(T, *, µ=µ_E):
    return np.cbrt(T**2 * µ / 4. / pi / pi)

# vis viva
def orbital_speed(r, a, *, µ=µ_E):
    return np.sqrt(µ * (2 / r - 1 / a))

def orbital_period(a,*,µ=µ_E):
    return 2 * pi * np.sqrt(a**3/µ)

def mean_motion(a,*,µ=µ_E):
    return np.sqrt(µ/(a**3))

def escape_velocity(r, µ):
    return np.sqrt(2) * orbital_speed_circle(r, µ=µ)

def earth_orbital_velocity(h):
    return orbital_speed_circle(h + r_E, µ=µ_E)

def earth_orbital_period(h):
    return orbital_period_circle(h + r_E, µ=µ_E)

def relative_area_visible_from_height(h, *, R=r_E):
    r = h+R
    phi = np.arccos(R / r)
    mantel_area = 2 * pi * R**2 * (1 - np.cos(phi))
    area_whole = 4 * pi * R**2
    return mantel_area/area_whole

def hohmann_manoeuvre(h1, h2, *, R=r_E, µ=µ_E):
    r1 = h1 + R
    r2 = h2 + R
    a = (r1+r2) / 2
    v1 = orbital_speed_circle(r1, µ=µ)
    v2_p = orbital_speed(r1, a, µ=µ)
    v2_a = orbital_speed(r2, a, µ=µ)
    v3 = orbital_speed_circle(r2, µ=µ)
    dV = abs(v2_p - v1) + abs(v3 - v2_a)
    return dV

def inclination_change(i, v1, v2 = None):
    if (v2 is None):
        return 2 * v1 * np.sin(i / 2)
    return sqrt(v1**2 + v2**2 - 2 * v1 * v2 * np.cos(i))

def hohmann_and_inclination_change(h1, h2, i, *, R=r_E, µ=µ_E):
    # Inclination change in the second burn of the Hoffmann manoeuvre
    r1 = h1 + R
    r2 = h2 + R
    a = (r1+r2) / 2
    v1 = orbital_speed_circle(r1, µ=µ)
    v2_p = orbital_speed(r1, a, µ=µ)
    v2_a = orbital_speed(r2, a, µ=µ)
    v3 = orbital_speed_circle(r2, µ=µ)
    dV = abs(v2_p - v1) + inclination_change(i, v2_a, v3)
    return dV

def hohmann_time(h1,h2,*,R=r_E,µ=µ_E):
    return orbital_period((h1+h2)/2+r_E, µ=µ) / 2

def bi_elliptical_transfer(h1,h2, h_util, *,R=r_E,µ=µ_E):
    rA = h1 + R
    rB = h_util + R
    rC = h2 + h_util

    vA = orbital_speed_circle(rA, µ=µ)
    vAB = orbital_speed(rA, (rA+rB)/2, µ=µ)
    dvAB = vAB - vA

    vBC = orbital_speed(rB, (rB+rC)/2, µ=µ)
    dvBC = orbital_speed(rB, (rB+rC)/2, µ=µ) - orbital_speed(rB, (rA+rB)/2, µ=µ)

    vCD = orbital_speed_circle(rC, µ=µ)
    dvCD = orbital_speed(rC, (rB+rC)/2, µ=µ) - orbital_speed_circle(rC, µ=µ)

    return dvAB + dvBC + dvCD

def bi_elliptical_time(h1,h2, h_util, *, R=r_E, µ=µ_E):
    T1u = orbital_period((h1+h_util)/2+R, µ=µ) / 2
    Tu2 = orbital_period((h2+h_util)/2+R, µ=µ) / 2
    return T1u + Tu2

def bi_elliptical_and_inclination_change(h1,h2, h_util, di, *, R=r_E, µ=µ_E):
    rA = h1 + R
    rB = h_util + R
    rC = h2 + h_util

    vA = orbital_speed_circle(rA, µ=µ)
    vAB = orbital_speed(rA, (rA+rB)/2, µ=µ)
    dvAB = vAB - vA

    vBCa = orbital_speed(rB,(rA+rB)/2, µ=µ)
    vBC = orbital_speed(rB,(rB+rC)/2, µ=µ)
    dvBC = inclination_change(di, vBCa, vBC)

    vCD = orbital_speed_circle(rC, µ=µ)
    dvCD = orbital_speed(rC,(rB+rC)/2, µ=µ) - orbital_speed_circle(rC, µ=µ)

    return dvAB + dvBC + dvCD

def J2_perturbation(a, e, i, *, J2=J2_E):
    return -3/2 * mean_motion(a) * J2 * (r_E/a)**2 * np.cos(i) / (1-e**2)**2

def J2_inclination_for_period(a, e, T, *, J2=J2_E):
    precession = 2*pi / T
    return np.arccos(-2 * precession * a**2 / (3 * mean_motion(a) * J2 * r_E**2))

