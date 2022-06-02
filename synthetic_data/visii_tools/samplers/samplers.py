# Standard Library
import itertools
import math
import random

# Third Party
import numpy as np
import transforms3d


def positon_from_radius_elevation_azimuth(radius, elevation, azimuth, degrees=True):
    """
    These are assumed to be in degrees

    """

    if degrees:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)

    yaw = azimuth
    roll = elevation

    R_azimuth = transforms3d.euler.euler2mat(0, 0, yaw)
    R_elevation = transforms3d.euler.euler2mat(roll, 0, 0)
    # R = transforms3d.euler.euler2mat(roll, pitch, yaw, 'sxyz')
    vec = np.array([0, 1, 0])
    vec = R_azimuth @ R_elevation @ vec
    pos = vec * radius

    return pos


def hemisphere_grid_sampler(radius_min=None,
                            radius_max=None,
                            radius_step=None,
                            azimuth_min=None,
                            azimuth_max=None,
                            azimuth_step=None,
                            elevation_min=None,
                            elevation_max=None,
                            elevation_step=None,
                            center=None,
                            verbose=False):
    """
    Samples positions on a hemisphere centered at center

    Azimuth andd elevation are in degrees
    """

    if center is None:
        center = [0, 0, 0]

    center = np.array(center)

    positions = []  # list of numpy arrays

    n = math.ceil((radius_max - radius_min) / radius_step) + 1
    print("n radius", n)
    radius_vals = np.linspace(radius_min, radius_max, n, endpoint=True)

    n = math.ceil((azimuth_max - azimuth_min) / azimuth_step) + 1
    azimuth_vals = np.linspace(azimuth_min, azimuth_max, n, endpoint=True)

    n = math.ceil((elevation_max - elevation_min) / elevation_step) + 1
    elevation_vals = np.linspace(
        elevation_min, elevation_max, n, endpoint=True)

    if verbose:
        print("radius samples", radius_vals)
        print("azimuth samples", azimuth_vals)
        print("elevation samples", elevation_vals)

    for radius, elevation, azimuth in itertools.product(radius_vals, elevation_vals, azimuth_vals):
        pos = positon_from_radius_elevation_azimuth(
            radius, elevation, azimuth) + center

        positions.append(pos)

    return positions


def hemisphere_random_sampler(radius_min=None,
                              radius_max=None,
                              azimuth_min=None,
                              azimuth_max=None,
                              elevation_min=None,
                              elevation_max=None,
                              num_samples=None,
                              degrees=True,
                              center=0):
    """
    Randomly samples positions on a hemisphere

    Azimuth ahd elevation are in degrees
    """

    positions = []
    for i in range(num_samples):
        radius = random.uniform(radius_min, radius_max)
        elevation = random.uniform(elevation_min, elevation_max)
        azimuth = random.uniform(azimuth_min, azimuth_max)

        pos = positon_from_radius_elevation_azimuth(
            radius, elevation, azimuth, degrees=degrees) + center
        positions.append(pos)

    return positions
