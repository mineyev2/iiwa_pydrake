# General imports
import argparse
import queue
import threading

import matplotlib

matplotlib.use("Agg")

from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

# Personal files
from demo_config import get_config

# Drake imports
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from mpl_toolkits.mplot3d import Axes3D
from pydrake.all import (
    AddFrameTriadIllustration,
    ApplySimulatorConfig,
    Box,
    ConstantVectorSource,
    DiagramBuilder,
    FrameIndex,
    LogVectorOutput,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
)
from termcolor import colored

from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.util.traj_planning import create_traj_from_q1_to_q2
from utils.kuka_geo_kin import KinematicsSolver
from utils.plotting import (
    plot_hemisphere_trajectory,
    plot_hemisphere_waypoints,
    plot_optical_axis_trajectory,
    plot_path_with_frames,
    plot_trajectories_side_by_side,
    save_trajectory_plots,
)


class State(Enum):
    IDLE = auto()
    WAITING_FOR_NEXT_SCAN = auto()
    PLANNING_MOVE_TO_START = auto()
    MOVING_TO_START = auto()
    MOVING_ALONG_HEMISPHERE = auto()
    MOVING_DOWN_OPTICAL_AXIS = auto()
    COMPUTING_IKS = auto()
    PAUSE = auto()
    DONE = auto()


def hemisphere_slerp(A, B, center, radius, speed_factor=1.0):
    """
    Interpolate along the shortest path on a hemisphere between points A and B.

    Parameters:
        A, B: np.array, shape (3,) - start and end points on the hemisphere
        center: np.array, shape (3,) - center of the sphere
        radius: float - radius of the hemisphere
        num_points: int - number of points along the path
        hemisphere_axis: int - axis index defining the hemisphere (default 2 -> z>=0)
        speed_factor: float - factor to scale the speed of traversal along the path (default 0.5 for half speed)

    Returns:
        path: np.array, shape (num_points, 3) - interpolated points on hemisphere
    """
    # Shift to sphere-centered coordinates
    a = A - center
    b = B - center

    # Normalize to unit sphere
    a_hat = a / np.linalg.norm(a)
    b_hat = b / np.linalg.norm(b)

    # Compute angle between vectors
    dot = np.clip(np.dot(a_hat, b_hat), -1.0, 1.0)
    theta = np.arccos(dot)

    # Create time array for PiecewisePolynomial
    # Make t depend on the length of the arc that we are traversing for more natural timing (longer paths take more time)
    arc_length = radius * theta
    t_final = (
        arc_length * 125 / speed_factor
    )  # Scale by speed factor to make it faster or slower
    num_points = int(t_final * 500)  # 500Hz update rate seems good

    t = np.linspace(0, t_final, num_points)

    # if theta < 1e-6:  # points are extremely close
    #     path = np.tile(A, (num_points, 1))
    #     return path

    # Slerp interpolation
    t_vals = np.linspace(0, 1, num_points)
    path = np.zeros((num_points, 3))
    for i, t_val in enumerate(t_vals):
        path[i] = (
            np.sin((1 - t_val) * theta) * a_hat + np.sin(t_val * theta) * b_hat
        ) / np.sin(theta)

    # Scale and shift back to original sphere
    path = center + radius * path

    # # Enforce hemisphere constraint
    # path[:, hemisphere_axis] = np.maximum(path[:, hemisphere_axis], center[hemisphere_axis])

    return path.T, t


def sphere_frame(p, hemisphere_axis, center):
    """
    Compute a smooth end-effector rotation matrix at point p on a sphere.

    z-axis  -> surface normal
    x-axis  -> projected global reference direction (smooth, no twisting)
    y-axis  -> z cross x

    Parameters
    ----------
    p : array-like (3,)
        Point on the sphere.
    center : array-like (3,)
        Sphere center (default origin).

    Returns
    -------
    R : (3,3) numpy array
        Rotation matrix with columns [x, y, z]
    """

    p = np.asarray(p, dtype=float)
    center = np.asarray(center, dtype=float)

    # Surface normal
    z = p - center
    z_norm = np.linalg.norm(z)
    if z_norm < 1e-9:
        raise ValueError("Point cannot equal sphere center.")
    z = z / z_norm

    # Global reference direction (choose something stable)
    if np.allclose(hemisphere_axis, np.array([1, 0, 0])):
        g = np.array([-1.0, 0.0, 0.0])
        if np.dot(z, g) > 0.99:
            g = np.array([0.0, 0.0, -1.0])
        elif np.dot(z, g) < -0.99:
            g = np.array([0.0, 0.0, 1.0])
    elif np.allclose(hemisphere_axis, np.array([-1, 0, 0])):
        g = np.array([0.0, 0.0, 1.0])
        if np.dot(z, g) > 0.99:
            g = np.array([0.0, 0.0, -1.0])
        elif np.dot(z, g) < -0.99:
            g = np.array([0.0, 0.0, 1.0])
    elif np.allclose(hemisphere_axis, np.array([0, 0, 1])):
        g = np.array([1.0, 0.0, 0.0])
        if np.dot(z, g) > 0.99:
            g = np.array([0.0, 0.0, -1.0])
        elif np.dot(z, g) < -0.99:
            g = np.array([0.0, 0.0, 1.0])
    elif np.allclose(hemisphere_axis, np.array([0, 0, -1])):  # NOTE: Haven't checked
        g = np.array([1.0, 0.0, 0.0])
        if np.dot(z, g) > 0.99:
            g = np.array([0.0, 0.0, 1.0])
        elif np.dot(z, g) < -0.99:
            g = np.array([0.0, 0.0, -1.0])

    # Project g onto tangent plane
    x = g - np.dot(g, z) * z
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-9:
        raise ValueError("Degenerate tangent direction.")
    x = x / x_norm

    # Complete right-handed frame
    y = np.cross(z, x)

    # Ensure orthonormality (numerical cleanup)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)

    R = np.column_stack((x, y, z))

    return R


def generate_hemisphere_waypoints(
    center,
    radius,
    hemisphere_axis,
    coverage=0.2,
    num_scan_points=30,
):
    """
    Generate N approximately uniformly distributed waypoints on a hemisphere.

    Args:
        center: (3,) array of hemisphere center
        radius: float, hemisphere radius
        hemisphere_axis: np.array of shape (3,), axis defining the hemisphere (e.g. [1, 0, 0] for x-axis hemisphere)
        coverage: float between 0 and 1, fraction of hemisphere to cover (default 0.5 for half hemisphere)
        num_scan_points: int, number of waypoints
    """

    waypoints = []
    phi_golden = (1 + np.sqrt(5)) / 2  # golden ratio

    # Normalize hemisphere_axis
    hemisphere_axis = hemisphere_axis / np.linalg.norm(hemisphere_axis)

    for k in range(num_scan_points):
        # Generate points on canonical hemisphere (top at [0, 0, 1])
        z_s = (
            1 - k / (num_scan_points - 1) * coverage
        )  # height from top (1) to equator (0)
        r_xy = np.sqrt(1 - z_s**2)  # radius in xy-plane
        theta = 2 * np.pi * k / phi_golden  # golden angle

        x_s = r_xy * np.cos(theta)
        y_s = r_xy * np.sin(theta)

        # Point on unit hemisphere with top at [0, 0, 1]
        point_canonical = np.array([x_s, y_s, z_s])

        # Compute rotation from [0, 0, 1] to hemisphere_axis
        z_ref = np.array([0.0, 0.0, 1.0])

        # If hemisphere_axis is close to [0, 0, 1], no rotation needed
        if np.allclose(hemisphere_axis, z_ref):
            point_rotated = point_canonical
        # If hemisphere_axis is close to [0, 0, -1], rotate 180 deg around x-axis
        elif np.allclose(hemisphere_axis, -z_ref):
            point_rotated = np.array(
                [point_canonical[0], -point_canonical[1], -point_canonical[2]]
            )
        else:
            # Compute rotation axis (perpendicular to both vectors)
            rotation_axis = np.cross(z_ref, hemisphere_axis)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # Compute rotation angle
            cos_angle = np.dot(z_ref, hemisphere_axis)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

            # Rodrigues' rotation formula
            K = np.array(
                [
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0],
                ]
            )
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

            point_rotated = R @ point_canonical

        # Scale by radius and translate by center
        point_world = center + radius * point_rotated

        # add custom rotation
        additional_rotation = RotationMatrix(
            np.array(
                [
                    [np.cos(-np.pi / 2), -np.sin(-np.pi / 2), 0],
                    [np.sin(-np.pi / 2), np.cos(-np.pi / 2), 0],
                    [0, 0, 1],
                ]
            )
        )  # -90 deg rotation around z-axis

        rotation = RotationMatrix(sphere_frame(point_world, hemisphere_axis, center))
        # rotation = additional_rotation @ rotation
        waypoint = RigidTransform(rotation, point_world)
        waypoints.append(waypoint)

    return waypoints


def generate_poses_along_hemisphere(
    center, radius, pose_curr, pose_target, hemisphere_axis, speed_factor=1.0
):
    """
    Args:
        - center: (3,) array of hemisphere center
        - radius: radius of hemisphere
        - pose_curr: RigidTransform of current end-effector pose
        - pose_target: RigidTransform of desired end-effector pose on hemisphere
    Returns:
        - path_points: (3, N) array of positions along the path
        - path_rots: List of (3, 3) rotation matrices at each point along the path
    """

    # Step 1: Generate shortest path along hemisphere surface
    A = pose_curr.translation()
    B = pose_target.translation()
    path_points, t = hemisphere_slerp(A, B, center, radius, speed_factor=speed_factor)

    # Generate rotation matrices along the path using the sphere_frame function
    path_rots = []
    num_points = path_points.shape[1]
    for i in range(num_points):
        p = path_points[:, i]
        R = sphere_frame(p, hemisphere_axis, center)
        path_rots.append(R)

    return path_points, path_rots, t


def generate_waypoints_down_optical_axis(
    pose_curr: RigidTransform,
    num_points: int = 100,
    # t_final: float = 2,
    distance: float = 0.025,
    speed_factor: float = 1.0,
):
    """
    Generate waypoints along the optical axis of the current end-effector pose.

    Args:
        pose_curr: RigidTransform of current end-effector pose
        num_points: Number of points to generate along the optical axis
        t_final: Total time to traverse the path (for timing the trajectory)

    Returns:
        List of RigidTransform representing the waypoints
    """
    path_points = []
    path_rots = []

    # Make t_final relative to distance.
    t_final = (distance * 2 / 0.025) / speed_factor

    for i in range(num_points):
        # Move down the optical axis (negative z direction in end-effector frame)
        delta_z = (
            -distance * i / num_points
        )  # Move down 10 cm over the course of the path
        delta_transform = RigidTransform(
            np.array([0, 0, delta_z])
        )  # No rotation change, just translation down z-axis
        waypoint = pose_curr @ delta_transform  # Apply the delta to the current pose
        path_points.append(waypoint.translation())
        path_rots.append(waypoint.rotation().matrix())

    path_points = np.array(path_points).T  # Shape (3, num_points)

    return path_points, path_rots, np.linspace(0, t_final, num_points)


def find_target_pose_on_hemisphere(center, latitude_deg, longitude_deg, radius):
    """
    Given a hemisphere defined by its center and radius, find the target end-effector pose on the hemisphere surface corresponding to the specified latitude and longitude angles.

    Args:
        center: (x, y, z) coordinates of the hemisphere center
        latitude_deg: Latitude angle in degrees (-90 to 90)
        longitude_deg: Longitude angle in degrees (-180 to 180)
        radius: Radius of the hemisphere
    Returns:
        target_pose: A 4x4 homogeneous transformation matrix representing the desired end-effector pose
    """

    latitude_rad = np.deg2rad(latitude_deg)
    longitude_rad = np.deg2rad(longitude_deg)
    x = center[0] - radius * np.cos(latitude_rad) * np.cos(longitude_rad)
    y = center[1] - radius * np.cos(latitude_rad) * np.sin(longitude_rad)
    z = center[2] + radius * np.sin(latitude_rad)

    target_pos = np.array([x, y, z])

    target_rot = sphere_frame(target_pos, center)

    return target_rot, target_pos


def check_joint_limits(trajectory_joint_poses, joint_lower_limits, joint_upper_limits):
    """
    Check if all joint positions in a trajectory are within specified limits.

    Args:
        trajectory_joint_poses: (7, N) array of joint positions
        joint_lower_limits: (7,) array of lower joint limits
        joint_upper_limits: (7,) array of upper joint limits

    Returns:
        is_valid: bool, True if all joints are within limits
        violations: list of dicts containing violation info with keys:
                   'joint_idx', 'waypoint_idx', 'value', 'limit_type', 'limit_value'
    """
    violations = []
    num_joints, num_waypoints = trajectory_joint_poses.shape

    for i in range(num_joints):
        for j in range(num_waypoints):
            joint_value = trajectory_joint_poses[i, j]

            # Check lower limit
            if joint_value < joint_lower_limits[i]:
                violations.append(
                    {
                        "joint_idx": i,
                        "waypoint_idx": j,
                        "value": joint_value,
                        "limit_type": "lower",
                        "limit_value": joint_lower_limits[i],
                        "violation_amount": joint_lower_limits[i] - joint_value,
                    }
                )

            # Check upper limit
            if joint_value > joint_upper_limits[i]:
                violations.append(
                    {
                        "joint_idx": i,
                        "waypoint_idx": j,
                        "value": joint_value,
                        "limit_type": "upper",
                        "limit_value": joint_upper_limits[i],
                        "violation_amount": joint_value - joint_upper_limits[i],
                    }
                )

    is_valid = len(violations) == 0

    if not is_valid:
        print(colored(f"⚠ Found {len(violations)} joint limit violations:", "yellow"))
        for v in violations[:5]:  # Show first 5 violations
            print(
                colored(
                    f"  Joint {v['joint_idx']+1}, waypoint {v['waypoint_idx']}: "
                    f"{np.rad2deg(v['value']):.2f}° violates {v['limit_type']} limit "
                    f"{np.rad2deg(v['limit_value']):.2f}° by {np.rad2deg(v['violation_amount']):.2f}°",
                    "yellow",
                )
            )
        if len(violations) > 5:
            print(colored(f"  ... and {len(violations) - 5} more violations", "yellow"))

    return is_valid, violations


def check_joint_velocities(
    trajectory_joint_poses, t, max_joint_velocities=np.deg2rad(60 * np.ones(7))
):  # Example limits in rad/s
    """
    Check if joint velocities in a trajectory exceed specified limits.

    Args:
        trajectory_joint_poses: (7, N) array of joint positions
        t: (N,) array of time values
        max_joint_velocities: (7,) array of maximum allowed joint velocities (absolute value)

    Returns:
        is_valid: bool, True if all velocities are within limits
        violations: list of dicts containing violation info
        velocities: (7, N-1) array of computed joint velocities
    """
    violations = []
    num_joints, num_waypoints = trajectory_joint_poses.shape

    # Compute velocities using finite differences
    dt = np.diff(t)  # Shape (N-1,)
    dq = np.diff(trajectory_joint_poses, axis=1)  # Shape (7, N-1)
    velocities = dq / dt  # Shape (7, N-1)

    # store max joint velocity across all joints and waypoints for reporting
    max_recorded_velocity = 0.0

    for i in range(num_joints):
        for j in range(velocities.shape[1]):
            vel_abs = np.abs(velocities[i, j])
            max_recorded_velocity = max(max_recorded_velocity, vel_abs)

            if vel_abs > max_joint_velocities[i]:
                violations.append(
                    {
                        "joint_idx": i,
                        "waypoint_idx": j,
                        "velocity": velocities[i, j],
                        "velocity_abs": vel_abs,
                        "limit": max_joint_velocities[i],
                        "violation_amount": vel_abs - max_joint_velocities[i],
                    }
                )

    is_valid = len(violations) == 0

    if not is_valid:
        print(
            colored(f"⚠ Found {len(violations)} joint velocity violations:", "yellow")
        )
        for v in violations[:5]:  # Show first 5 violations
            print(
                colored(
                    f"  Joint {v['joint_idx']+1}, waypoint {v['waypoint_idx']}: "
                    f"velocity {np.rad2deg(v['velocity']):.2f}°/s (abs: {np.rad2deg(v['velocity_abs']):.2f}°/s) "
                    f"exceeds limit {np.rad2deg(v['limit']):.2f}°/s by {np.rad2deg(v['violation_amount']):.2f}°/s",
                    "yellow",
                )
            )
        if len(violations) > 5:
            print(colored(f"  ... and {len(violations) - 5} more violations", "yellow"))

    return is_valid, violations, max_recorded_velocity


def generate_IK_solutions_for_path(
    path_points,
    path_rots,
    kinematics_solver,
    q_init,
    elbow_angle,
    joint_lower_limits,
    joint_upper_limits,
):
    trajectory_joint_poses = []
    q_prev = (
        q_init  # Try to match first point to current joint configuration for smoothness
    )

    for i in range(len(path_points.T)):
        eef_pos = path_points[:, i]  # Shift spiral to be around the hemisphere center
        eef_rot = path_rots[i]  # Use the rotation matrix from the path
        Q = kinematics_solver.IK_for_microscope(eef_rot, eef_pos, psi=elbow_angle)

        q_curr = kinematics_solver.find_closest_solution(
            Q, q_prev
        )  # Choose closest solution to previous point for smoothness

        trajectory_joint_poses.append(q_curr)
        q_prev = q_curr

    trajectory_joint_poses = np.array(trajectory_joint_poses).T  # Shape (7, num_points)

    return trajectory_joint_poses


def compute_hemisphere_traj_async(
    hemisphere_pos,
    hemisphere_radius,
    hemisphere_axis,
    eef_pose,
    pose_target,
    kinematics_solver,
    q_curr,
    elbow_angle,
    ik_result,
    plot_trajectories=False,
    scan_idx=0,
    joint_lower_limits=None,
    joint_upper_limits=None,
    speed_factor=1.0,
    max_joint_velocities=None,
):
    hemisphere_points, hemisphere_rots, hemisphere_t = generate_poses_along_hemisphere(
        center=hemisphere_pos,
        radius=hemisphere_radius,
        pose_curr=eef_pose,
        pose_target=pose_target,
        hemisphere_axis=hemisphere_axis,
        speed_factor=speed_factor,
    )

    trajectory_joint_poses = generate_IK_solutions_for_path(
        path_points=hemisphere_points,
        path_rots=hemisphere_rots,
        kinematics_solver=kinematics_solver,
        q_init=q_curr,
        elbow_angle=elbow_angle,
        joint_lower_limits=joint_lower_limits,
        joint_upper_limits=joint_upper_limits,
    )

    # Turn into piecewise polynomial trajectory
    traj = PiecewisePolynomial.FirstOrderHold(hemisphere_t, trajectory_joint_poses)
    print(f"Trajectory start_time: {traj.start_time()}, end_time: {traj.end_time()}")

    # Store results (including raw data for plotting)
    ik_result["trajectory"] = traj
    ik_result["trajectory_joint_poses"] = trajectory_joint_poses
    ik_result["time"] = hemisphere_t
    ik_result["scan_idx"] = scan_idx

    # Generate and save hemisphere trajectory plot (non-blocking, thread-safe)
    if plot_trajectories:
        plot_hemisphere_trajectory(
            trajectory_joint_poses,
            hemisphere_t,
            scan_idx,
            joint_lower_limits,
            joint_upper_limits,
        )

    # Check to make sure velocities are within limits, if not, raise error to trigger replanning with slower speed
    velocities_are_valid, _, max_recorded_velocity = check_joint_velocities(
        trajectory_joint_poses, hemisphere_t, max_joint_velocities=max_joint_velocities
    )
    print(
        colored(
            f"Max recorded joint velocity in hemisphere trajectory: {np.rad2deg(max_recorded_velocity):.2f}°/s",
            "yellow",
        )
    )
    if not velocities_are_valid:
        print(
            colored(
                "❌ Joint velocity limit(s) exceeded in hemisphere trajectory", "red"
            )
        )
        ik_result["valid"] = False
        ik_result["ready"] = True
        return

    joints_are_valid, num_violations = check_joint_limits(
        trajectory_joint_poses, joint_lower_limits, joint_upper_limits
    )

    if not joints_are_valid:
        print(colored("❌ Joint limit(s) exceeded in hemisphere trajectory", "red"))
        ik_result["valid"] = False
        ik_result["ready"] = True
        return

    ik_result["valid"] = True
    ik_result["ready"] = True
    print(colored("✓ Hemisphere IK computation complete!", "green"))


def compute_optical_axis_traj_async(
    pose_curr,
    kinematics_solver,
    q_curr,
    elbow_angle,
    ik_result,
    plot_trajectories=False,
    scan_idx=0,
    joint_lower_limits=None,
    joint_upper_limits=None,
    distance: float = 0.025,
    speed_factor: float = 1.0,
    max_joint_velocities=None,
):
    print("q curr in compute_optical_axis_traj_async: ", np.rad2deg(q_curr).tolist())
    path_points, path_rots, t = generate_waypoints_down_optical_axis(
        pose_curr, distance=distance, speed_factor=speed_factor
    )

    trajectory_joint_poses = generate_IK_solutions_for_path(
        path_points=path_points,
        path_rots=path_rots,
        kinematics_solver=kinematics_solver,
        q_init=q_curr,
        elbow_angle=elbow_angle,
        joint_lower_limits=joint_lower_limits,
        joint_upper_limits=joint_upper_limits,
    )

    # # append trajectory to itself but reverese the waypoints to move back up the optical axis
    trajectory_joint_poses = np.hstack(
        (trajectory_joint_poses, trajectory_joint_poses[:, ::-1])
    )
    t = np.hstack((t, t + t[-1] + t[1]))  # Time for returning back up the optical axis

    traj = PiecewisePolynomial.FirstOrderHold(t, trajectory_joint_poses)

    # Store results (including raw data for plotting)
    ik_result["trajectory"] = traj
    ik_result["trajectory_joint_poses"] = trajectory_joint_poses
    ik_result["time"] = t
    ik_result["scan_idx"] = scan_idx

    # Generate and save optical axis trajectory plot (non-blocking, thread-safe)
    if plot_trajectories:
        plot_optical_axis_trajectory(
            trajectory_joint_poses,
            t,
            scan_idx,
            joint_lower_limits,
            joint_upper_limits,
        )

    # Check to make sure velocities are within limits, if not, raise error to trigger replanning with slower speed
    velocities_are_valid, _, max_recorded_velocity = check_joint_velocities(
        trajectory_joint_poses, t, max_joint_velocities=max_joint_velocities
    )

    print(
        colored(
            f"Max recorded joint velocity in optical axis trajectory: {np.rad2deg(max_recorded_velocity):.2f}°/s",
            "yellow",
        )
    )

    if not velocities_are_valid:
        print(
            colored(
                "❌ Joint velocity limit(s) exceeded in optical axis trajectory", "red"
            )
        )
        ik_result["valid"] = False
        ik_result["ready"] = True
        return

    joints_are_valid, num_violations = check_joint_limits(
        trajectory_joint_poses, joint_lower_limits, joint_upper_limits
    )

    if not joints_are_valid:
        print(colored("❌ Joint limit(s) exceeded in optical axis trajectory", "red"))
        ik_result["valid"] = False
        ik_result["ready"] = True
        return

    ik_result["valid"] = True
    ik_result["ready"] = True
    print(colored("✓ Optical axis IK computation complete!", "green"))


def main(use_hardware: bool, no_cam: bool = False) -> None:
    # Load configuration
    cfg = get_config(use_hardware)
    speed_factor = cfg["speed_factor"]
    max_joint_velocities = cfg["max_joint_velocities"]

    # Clean up trajectory output folders
    import shutil

    hemisphere_paths_dir = Path(__file__).parent.parent / "outputs" / "hemisphere_paths"
    optical_axis_paths_dir = (
        Path(__file__).parent.parent / "outputs" / "optical_axis_paths"
    )

    if hemisphere_paths_dir.exists():
        shutil.rmtree(hemisphere_paths_dir)
        print(colored(f"✓ Cleared {hemisphere_paths_dir}", "grey"))

    if optical_axis_paths_dir.exists():
        shutil.rmtree(optical_axis_paths_dir)
        print(colored(f"✓ Cleared {optical_axis_paths_dir}", "grey"))

    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    # - add_model:
    #     name: sphere_obstacle
    #     file: package://iiwa_setup/sphere_obstacle.sdf
    # - add_weld:
    #     parent: worldhemisphere_radius
    #     child: sphere_obstacle::sphere_body
    #     X_PC:
    #         translation: [0.5, 0.0, 0.6]
    plant_config:
        # For some reason, this requires a small timestep
        time_step: 0.005
        contact_model: "hydroelastic_with_fallback"
        discrete_contact_approximation: "sap"
    model_drivers:
        iiwa: !IiwaDriver
            lcm_bus: "default"
            control_mode: position_only
    lcm_buses:
        default:
            lcm_url: ""
    """

    hemisphere_paths_dir = Path(__file__).parent.parent / "outputs" / "hemisphere_paths"
    optical_axis_paths_dir = (
        Path(__file__).parent.parent / "outputs" / "optical_axis_paths"
    )

    if hemisphere_paths_dir.exists():
        shutil.rmtree(hemisphere_paths_dir)
        print(colored(f"✓ Cleared {hemisphere_paths_dir}", "grey"))

    if optical_axis_paths_dir.exists():
        shutil.rmtree(optical_axis_paths_dir)
        print(colored(f"✓ Cleared {optical_axis_paths_dir}", "grey"))

    # ==================================================================
    # Waypoint Generation Setup
    # ==================================================================
    hemisphere_pos = np.array([0.8, 0.0, 0.36])
    hemisphere_radius = 0.100
    hemisphere_axis = np.array([-1, 0, 0])
    num_scan_points = 50
    coverage = 0.40  # Fraction of hemisphere to cover
    distance_along_optical_axis = 0.025
    num_pictures = 30
    # elbow_angle = np.pi / 2 + np.deg2rad(45)
    # elbow_angle = np.pi / 2
    elbow_angle = np.pi / 2 - np.deg2rad(45)
    scan_idx = 35  # Default is 1

    r = np.array([0, 0, -1])
    v = np.array([0, 1, 0])

    T_tip_to_camera = np.eye(4)
    T_tip_to_camera[:3, 3] = [0, 0, 0.1]
    T_tip_to_camera[:3, :3] = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ]
    )

    # Generate waypoints
    hemisphere_waypoints = generate_hemisphere_waypoints(
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        num_scan_points=num_scan_points,
        coverage=coverage,
    )

    # Plot waypoints for sanity check
    heimsphere_waypoints_output_path = (
        Path(__file__).parent.parent / "outputs" / "hemisphere_waypoints.png"
    )
    plot_hemisphere_waypoints(
        hemisphere_waypoints,
        hemisphere_pos,
        hemisphere_radius,
        hemisphere_axis,
        output_path=heimsphere_waypoints_output_path,
        visualize=True,
    )

    # ===================================================================
    # Diagram Setup
    # ===================================================================
    builder = DiagramBuilder()

    # Load scenario
    scenario = LoadScenario(data=scenario_data)

    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_pos=hemisphere_pos,
            hemisphere_radius=hemisphere_radius,
            use_hardware=use_hardware,
        ),
    )

    # Log joint positions using station's exported output port
    from pydrake.systems.primitives import VectorLogSink

    state_logger = builder.AddSystem(VectorLogSink(7))
    state_logger.set_name("state_logger")
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"), state_logger.get_input_port()
    )

    # Create dummy constant position source (using station's default position)
    default_position = station.get_iiwa_controller_plant().GetPositions(
        station.get_iiwa_controller_plant().CreateDefaultContext()
    )
    dummy = builder.AddSystem(ConstantVectorSource(default_position))
    builder.Connect(
        dummy.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    # Visualize internal station with Meshcat
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    # Get total number of frames
    camera_frame = station.get_internal_plant().GetFrameByName("camera_link")
    AddFrameTriadIllustration(
        scene_graph=station.internal_station.get_scene_graph(),
        plant=station.get_internal_plant(),
        frame=camera_frame,
        length=0.1,
        radius=0.002,
        name="camera_link",
    )

    # Build diagram
    diagram = builder.Build()

    # ====================================================================
    # Simulator Setup
    # ====================================================================
    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    station.internal_meshcat.AddButton("Plan Move to Start")
    station.internal_meshcat.AddButton("Move to Start")
    station.internal_meshcat.AddButton("Execute Trajectory")

    # Add joint position sliders (in degrees for readability)
    joint_lower_limits = station.get_internal_plant().GetPositionLowerLimits()
    joint_upper_limits = station.get_internal_plant().GetPositionUpperLimits()

    for i in range(7):
        min_deg = np.rad2deg(joint_lower_limits[i])
        max_deg = np.rad2deg(joint_upper_limits[i])
        station.internal_meshcat.AddSlider(
            f"Joint {i+1} (deg)", min_deg, max_deg, 0.1, 0
        )

    # Add PSI angle slider (read-only visualization)
    station.internal_meshcat.AddSlider("Current PSI (deg)", -180, 180, 0.1, 0)

    # region Step 1) Solve IK for desired pose
    kinematics_solver = KinematicsSolver(station, r, v)

    station_context = station.GetMyContextFromRoot(simulator.get_context())

    # Calculate IK for getting to the top of the hemisphere

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    prev_state = State.IDLE
    state = State.IDLE

    joint_lower_limits = station.get_internal_plant().GetPositionLowerLimits()
    joint_upper_limits = station.get_internal_plant().GetPositionUpperLimits()

    # Button management
    num_move_to_top_clicks = 0
    num_compute_iks_clicks = 0
    num_execute_traj_clicks = 0

    # IK computation thread state
    ik_thread = None
    hemisphere_ik_result = {
        "ready": False,
        "valid": True,
        "trajectory": None,
        "trajectory_start_time": None,
    }
    optical_axis_ik_result = {
        "ready": False,
        "valid": True,
        "trajectory": None,
        "trajectory_start_time": None,
    }

    # Initialize SEW Plane visualization (Actual)
    station.internal_meshcat.SetObject(
        "sew_plane", Box(0.8, 0.8, 0.001), Rgba(0.0, 0.5, 1.0, 0.3)
    )
    # Initialize Reference Plane visualization (Psi = 0)
    station.internal_meshcat.SetObject(
        "ref_plane", Box(0.8, 0.8, 0.001), Rgba(1.0, 1.0, 1.0, 0.2)
    )

    # ---------------------------------------------------------------
    # Camera + two background threads
    #
    # _capture_thread  ←  owns camera.read() loop at camera's native rate.
    #                      Always stores the latest frame in _latest_frame.
    #                      Main loop NEVER calls camera.read().
    #
    # _writer_thread   ←  owns cv2.imwrite(); pulls from _frame_queue.
    #
    # At each pause the main loop just snapshots _latest_frame (a lock
    # grab, ~microseconds) and enqueues it → zero blocking on critical path.
    # ---------------------------------------------------------------
    camera = None
    _latest_frame = None
    _latest_frame_lock = None
    _capture_stop = None
    _frame_queue = None
    _capture_thread = None
    _writer_thread = None

    if not no_cam:
        camera = cv2.VideoCapture(4)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not camera.isOpened():
            print(
                colored(
                    "⚠ Could not open camera device 4 – frames will NOT be saved",
                    "yellow",
                )
            )

        _latest_frame_lock = threading.Lock()  # guards _latest_frame
        _capture_stop = threading.Event()  # set → capture thread exits
        _frame_queue: queue.Queue = queue.Queue(maxsize=0)  # unbounded

        def _capture_loop():
            """Runs in background; owns camera.read() entirely.
            Continuously updates _latest_frame so the main loop always has
            a fresh frame available without ever blocking on camera.read().
            Also keeps the camera driver buffer drained between pauses.
            """
            nonlocal _latest_frame
            while not _capture_stop.is_set():
                ret, frame = camera.read()  # blocks at camera rate (~33 ms/frame)
                if ret:
                    with _latest_frame_lock:
                        _latest_frame = frame  # overwrite; only the latest matters

        def _writer_loop():
            """Handles all cv2.imwrite() calls off the critical path."""
            while True:
                item = _frame_queue.get()
                if item is None:  # sentinel – exit
                    _frame_queue.task_done()
                    break
                frame_data, path_str = item
                cv2.imwrite(path_str, frame_data)
                _frame_queue.task_done()

        _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        _writer_thread = threading.Thread(target=_writer_loop, daemon=True)
        _capture_thread.start()
        _writer_thread.start()
        print(colored("✓ Camera threads started", "cyan"))
    else:
        print(colored("✓ Camera disabled via --no_cam", "yellow"))

    # Per-scan metadata
    scan_frame_dir = None  # Path to current scan's frame folder
    optical_halfway_time = None  # traj_time at which inward pass ends

    # Per-scan stop-and-shoot state (initialised in COMPUTING_IKS)
    capture_traj_times: np.ndarray = np.array([])  # traj_times at each photo stop
    next_capture_idx: int = 0  # which stop we're waiting to reach
    scan_frame_idx: int = 0  # frame counter within the scan
    is_pausing_for_capture: bool = False
    pause_start_sim_time: float = 0.0
    hold_traj_time: float = 0.0  # trajectory time to hold at during pause

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        if state != prev_state:
            print(colored(f"State changed: {prev_state} -> {state}", "grey"))
            prev_state = state

        # ------------------------------------------------------------------
        # Update SEW (Shoulder-Elbow-Wrist) Plane Visualization
        # ------------------------------------------------------------------
        station_context = station.GetMyContextFromRoot(simulator.get_context())
        internal_plant = station.get_internal_plant()
        internal_plant_context = station.get_internal_plant_context()

        # Joint 2, 4, 6 frames
        p_J2 = (
            internal_plant.GetFrameByName("iiwa_link_2")
            .CalcPoseInWorld(internal_plant_context)
            .translation()
        )
        p_J4 = (
            internal_plant.GetFrameByName("iiwa_link_4")
            .CalcPoseInWorld(internal_plant_context)
            .translation()
        )
        p_J6 = (
            internal_plant.GetFrameByName("iiwa_link_6")
            .CalcPoseInWorld(internal_plant_context)
            .translation()
        )

        # Vectors for plane math
        v24 = p_J4 - p_J2
        v26 = p_J6 - p_J2
        normal = np.cross(v26, v24)
        norm_val = np.linalg.norm(normal)

        if norm_val > 1e-4:
            normal = normal / norm_val
            # Use v24 as the local x-axis for stability
            x_axis = v24 / np.linalg.norm(v24)
            y_axis = np.cross(normal, x_axis)
            R_WP = RotationMatrix(np.column_stack([x_axis, y_axis, normal]))
            centroid = (p_J2 + p_J4 + p_J6) / 3.0
            station.internal_meshcat.SetTransform(
                "sew_plane", RigidTransform(R_WP, centroid)
            )

        # Update Reference Plane (using r, v from outer scope)
        p_SW = p_J6 - p_J2
        norm_SW = np.linalg.norm(p_SW)
        if norm_SW > 1e-4:
            e_SW = p_SW / norm_SW
            kr = np.cross(e_SW - r, v)
            norm_kr = np.linalg.norm(kr)
            if norm_kr > 1e-4:
                kr_unit = kr / norm_kr
                kx = np.cross(kr_unit, e_SW)
                norm_kx = np.linalg.norm(kx)
                if norm_kx > 1e-4:
                    ex = kx / norm_kx
                    ref_normal = np.cross(e_SW, ex)
                    R_WR = RotationMatrix(np.column_stack([e_SW, ex, ref_normal]))
                    station.internal_meshcat.SetTransform(
                        "ref_plane", RigidTransform(R_WR, (p_J2 + p_J6) / 2.0)
                    )

                    # Calculate and print PSI angle in real-time
                    # Use the cross/dot logic from SEWStereo.fwd_kin for the signed angle
                    n_hat_sew = normal
                    n_hat_ref = ref_normal

                    cross_dot = np.dot(n_hat_sew, np.cross(e_SW, n_hat_ref))
                    dot_product = np.dot(n_hat_sew, n_hat_ref)
                    psi_rad = np.arctan2(cross_dot, dot_product)

                    # Update PSI slider in real-time
                    station.internal_meshcat.SetSliderValue(
                        "Current PSI (deg)", np.rad2deg(psi_rad)
                    )
        # ------------------------------------------------------------------

        if state == State.IDLE:
            if station.internal_meshcat.GetButtonClicks("Plan Move to Start") > 0:
                print(colored("Planning move to start", "cyan"))

                station_context = station.GetMyContextFromRoot(simulator.get_context())
                q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                    station_context
                )

                target_pose = hemisphere_waypoints[scan_idx - 1]
                target_rot = target_pose.rotation().matrix()
                target_pos = target_pose.translation()

                Q = kinematics_solver.IK_for_microscope(
                    target_rot, target_pos, psi=elbow_angle
                )

                # # print all sols, round to nearest 2nd decimal
                # for i in range(Q.shape[0]):
                #     print(np.rad2deg(Q[i, :]).round(2))

                # Step 2) Find IK closest to current joint values
                station_context = station.GetMyContextFromRoot(simulator.get_context())
                q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                    station_context
                )
                q_des = kinematics_solver.find_closest_solution(Q, q_curr)

                # print(
                #     colored(
                #         f"Goal joint configuration for Start: {np.rad2deg(q_des)}",
                #         "yellow",
                #     )
                # )

                trajectory = create_traj_from_q1_to_q2(
                    station,
                    q_current,
                    q_des,
                )

                state = State.PLANNING_MOVE_TO_START
        elif state == State.PLANNING_MOVE_TO_START:
            if station.internal_meshcat.GetButtonClicks("Move to Start") > 0:
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_TO_START
        elif state == State.WAITING_FOR_NEXT_SCAN:
            # if (
            #     station.internal_meshcat.GetButtonClicks("Compute IKs")
            #     > num_compute_iks_clicks
            # ):
            #     num_compute_iks_clicks = station.internal_meshcat.GetButtonClicks(
            #         "Compute IKs"
            #     )
            # else:
            #     continue

            if scan_idx >= len(hemisphere_waypoints):
                print(colored("✓ All scans complete!", "green"))
                state = State.DONE
                continue

            print(colored(f"Preparing trajectory for scan #{scan_idx}", "grey"))

            pose_curr = hemisphere_waypoints[scan_idx - 1]  # We assume scan_idx >= 1
            pose_target = hemisphere_waypoints[scan_idx]

            # Get current end-effector pose from actual robot state
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            internal_plant = station.get_internal_plant()
            internal_plant_context = station.get_internal_plant_context()
            eef_pose = internal_plant.GetFrameByName(
                "microscope_tip_link"
            ).CalcPoseInWorld(internal_plant_context)
            q_curr = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            # Setup trajectory IK computations in background threads to avoid blocking the main simulation loop
            print(colored("Starting IK computation in background...", "yellow"))
            hemisphere_ik_result["ready"] = False
            hemisphere_ik_thread = threading.Thread(
                target=compute_hemisphere_traj_async,
                args=(
                    hemisphere_pos,
                    hemisphere_radius,
                    hemisphere_axis,
                    eef_pose,
                    pose_target,
                    kinematics_solver,
                    q_curr,
                    elbow_angle,
                    hemisphere_ik_result,
                    True,  # plot_trajectories
                    scan_idx,
                    joint_lower_limits,
                    joint_upper_limits,
                    speed_factor,
                    max_joint_velocities,
                ),
                daemon=True,
            )
            hemisphere_ik_thread.start()

            optical_axis_ik_result["ready"] = False
            optical_axis_ik_thread = threading.Thread(
                target=compute_optical_axis_traj_async,
                args=(
                    pose_curr,
                    kinematics_solver,
                    q_curr,
                    elbow_angle,
                    optical_axis_ik_result,
                    True,  # plot_trajectories
                    scan_idx,
                    joint_lower_limits,
                    joint_upper_limits,
                    distance_along_optical_axis,
                    speed_factor,
                    max_joint_velocities,
                ),
                daemon=True,
            )
            optical_axis_ik_thread.start()

            state = State.COMPUTING_IKS

        elif state == State.COMPUTING_IKS:
            # Wait for IK computation to complete
            if hemisphere_ik_result["ready"] and optical_axis_ik_result["ready"]:
                # if not hemisphere_ik_result["valid"] or not optical_axis_ik_result["valid"]:
                #     print(colored("❌ IK computation failed. Quitting program.", "red"))
                #     break

                hemisphere_trajectory = hemisphere_ik_result["trajectory"]
                optical_axis_trajectory = optical_axis_ik_result["trajectory"]

                # Plotting already happened in the async functions (non-blocking)

                trajectory_start_time = simulator.get_context().get_time()
                print(
                    f"Simulator time when starting trajectory: {trajectory_start_time}"
                )
                print(colored("Starting trajectory execution", "yellow"))

                # ----------------------------------------------------------
                # Compute the halfway point of the optical axis trajectory
                # (the trajectory is mirrored: first half goes inward, second
                # half goes outward). Photos are taken only on the inward pass.
                # ----------------------------------------------------------
                optical_halfway_time = optical_axis_trajectory.end_time() / 2.0

                # Create per-scan output folder: outputs/scans/scan<NN>/
                scans_base = Path(__file__).parent.parent / "outputs" / "scans"
                scan_frame_dir = scans_base / f"scan{scan_idx:02d}"
                scan_frame_dir.mkdir(parents=True, exist_ok=True)
                midpoint_pose_saved = False
                print(colored(f"✓ Frame output dir: {scan_frame_dir}", "cyan"))

                # ----------------------------------------------------------
                # Build the stop-and-shoot schedule.
                # num_pictures evenly-spaced stops on [0, optical_halfway_time];
                # both endpoints (t=0 = start, t=halfway = deepest point) included.
                # ----------------------------------------------------------
                capture_traj_times = np.linspace(
                    0.0, optical_halfway_time, num_pictures
                )
                next_capture_idx = 0
                scan_frame_idx = 0
                is_pausing_for_capture = False
                pause_start_sim_time = 0.0
                hold_traj_time = 0.0
                print(
                    colored(
                        f"✓ Stop-and-shoot: {num_pictures} stops at "
                        f"{np.round(capture_traj_times, 2).tolist()}",
                        "cyan",
                    )
                )

                scan_idx += 1

                # state = State.MOVING_DOWN_OPTICAL_AXIS
                state = State.MOVING_ALONG_HEMISPHERE

        elif state == State.MOVING_TO_START:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= trajectory.end_time():
                q_desired = trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Trajectory execution complete!", "green"))
                if scan_idx >= len(hemisphere_waypoints):
                    print(colored("✓ All scans complete!", "green"))
                    state = State.DONE
                else:
                    state = State.WAITING_FOR_NEXT_SCAN

        elif state == State.MOVING_ALONG_HEMISPHERE:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if traj_time <= hemisphere_trajectory.end_time():
                q_desired = hemisphere_trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )
            else:
                print(colored("✓ Hemisphere trajectory execution complete!", "green"))
                state = State.WAITING_FOR_NEXT_SCAN

        elif state == State.MOVING_DOWN_OPTICAL_AXIS:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time

            if is_pausing_for_capture:
                # -------------------------------------------------------
                # PAUSED: hold robot at stop position while timer runs.
                # After 0.5 s: flush camera buffer, take one still, queue
                # it for async write, compensate trajectory_start_time.
                # -------------------------------------------------------
                q_desired = optical_axis_trajectory.value(hold_traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )

                elapsed_pause = current_time - pause_start_sim_time
                if elapsed_pause >= 0.5:
                    # Grab the latest frame the capture thread has buffered (if camera enabled).
                    # This is a lock snapshot (~microseconds) – no blocking read.
                    if not no_cam:
                        with _latest_frame_lock:
                            frame = (
                                _latest_frame.copy()
                                if _latest_frame is not None
                                else None
                            )
                        if frame is not None and scan_frame_dir is not None:
                            frame_path = str(
                                scan_frame_dir / f"frame_{scan_frame_idx:05d}.jpg"
                            )
                            _frame_queue.put((frame, frame_path))
                            scan_frame_idx += 1
                            print(
                                colored(
                                    f"  📷 Photo {scan_frame_idx}/{num_pictures} "
                                    f"at traj_t={hold_traj_time:.3f}s "
                                    f"→ frame_{scan_frame_idx-1:05d}.jpg",
                                    "cyan",
                                )
                            )

                    # Shift trajectory_start_time so the robot resumes
                    # from exactly the point it paused at.
                    trajectory_start_time += elapsed_pause
                    next_capture_idx += 1
                    is_pausing_for_capture = False

            elif traj_time <= optical_axis_trajectory.end_time():
                q_desired = optical_axis_trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(
                    station_context, q_desired
                )

                # -----------------------------------------------------------
                # Check whether we've reached the next photo stop.
                # All stops are on the inward pass [0, optical_halfway_time].
                # -----------------------------------------------------------
                if (
                    next_capture_idx < len(capture_traj_times)
                    and traj_time >= capture_traj_times[next_capture_idx]
                ):
                    hold_traj_time = capture_traj_times[next_capture_idx]
                    pause_start_sim_time = current_time
                    is_pausing_for_capture = True
                    print(
                        colored(
                            f"  ⏸ Pausing at stop {next_capture_idx + 1}/{num_pictures} "
                            f"(traj_t={hold_traj_time:.3f}s)",
                            "yellow",
                        )
                    )

                # -----------------------------------------------------------
                # Save camera_link pose at the deepest inward point
                # (= optical_halfway_time, the last capture stop).
                # Saved once per scan on first crossing.
                # -----------------------------------------------------------
                if (
                    not midpoint_pose_saved
                    and scan_frame_dir is not None
                    and traj_time >= optical_halfway_time
                ):
                    internal_plant = station.get_internal_plant()
                    internal_plant_context = station.get_internal_plant_context()
                    camera_pose = internal_plant.GetFrameByName(
                        "camera_link"
                    ).CalcPoseInWorld(internal_plant_context)
                    pose_matrix = (
                        camera_pose.GetAsMatrix4()
                    )  # 4x4 homogeneous transform
                    pose_path = scan_frame_dir / "pose.npy"
                    np.save(str(pose_path), pose_matrix)
                    midpoint_pose_saved = True
                    print(
                        colored(
                            f"✓ Saved camera_link pose at deepest point "
                            f"t={traj_time:.3f}s → {pose_path}",
                            "cyan",
                        )
                    )

            elif traj_time <= optical_axis_trajectory.end_time() + 1:  # 1 s buffer
                pass
            else:
                print(colored("✓ Optical axis trajectory execution complete!", "green"))
                trajectory_start_time = simulator.get_context().get_time()
                state = State.MOVING_ALONG_HEMISPHERE

        elif state == State.DONE:
            context = simulator.get_context()
            log = state_logger.FindLog(context)
            t = log.sample_times()
            data = log.data()  # shape: (num_states, num_samples)

            # stack time as first column
            out = np.vstack((t, data)).T

            joint_log_csv_path = (
                Path(__file__).parent.parent / "outputs" / "joint_log.csv"
            )
            np.savetxt(
                joint_log_csv_path,
                out,
                delimiter=",",
                header="time," + ",".join([f"x{i}" for i in range(data.shape[0])]),
                comments="",
            )

            # Also save as pickle
            log_data = {"sample_times": log.sample_times(), "data": log.data()}
            import pickle

            joint_log_pkl_path = (
                Path(__file__).parent.parent / "outputs" / "joint_log.pkl"
            )
            with open(joint_log_pkl_path, "wb") as f:
                pickle.dump(log_data, f)
            break
            # pass

        # Update button counts
        num_move_to_top_clicks = station.internal_meshcat.GetButtonClicks(
            "Move to Start"
        )

        # num_compute_iks_clicks = station.internal_meshcat.GetButtonClicks(
        #     "Compute IKs"
        # )

        num_execute_traj_clicks = station.internal_meshcat.GetButtonClicks(
            "Execute Trajectory"
        )

        # Update joint sliders to show current joint positions
        station_context = station.GetMyContextFromRoot(simulator.get_context())
        q_current = station.GetOutputPort("iiwa.position_measured").Eval(
            station_context
        )
        for i in range(7):
            station.internal_meshcat.SetSliderValue(
                f"Joint {i+1} (deg)", np.rad2deg(q_current[i])
            )

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.01)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Plan Move to Start")
    station.internal_meshcat.DeleteButton("Move to Start")
    station.internal_meshcat.DeleteButton("Execute Trajectory")

    # Delete joint sliders
    for i in range(7):
        station.internal_meshcat.DeleteSlider(f"Joint {i+1} (deg)")

    # Shut down background threads cleanly.
    if not no_cam and camera is not None:
        # 1. Stop the capture thread (exits on next camera.read() return)
        _capture_stop.set()
        _capture_thread.join(timeout=5)
        camera.release()
        # 2. Drain and stop the writer thread
        _frame_queue.put(None)  # sentinel: tells writer thread to exit
        _frame_queue.join()  # wait for all queued imwrite()s to complete
        _writer_thread.join(timeout=10)
        print(colored("✓ Frame writer flushed and camera released", "grey"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--no_cam",
        action="store_true",
        help="Disable camera threads (no frames will be saved).",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware, no_cam=args.no_cam)
