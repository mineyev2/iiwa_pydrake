"""
Hemisphere scanning solver

Authors: Roman Mineyev
"""

import csv

# Other
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.transform import Rotation
from termcolor import colored

# Drake
from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram

# Personal files
from scripts.kuka_geo_kin import KinematicsSolver


class Node:
    """
    Node class

    Args:
        q (np.array): Joint configuration
        pose (np.array): End-effector pose as transformation matrix
        layer_idx (int): Index of the layer this node belongs to
        node_idx (int): Index of the node within its layer
    """

    def __init__(
        self, q, target_pos, eef_pos_distance, eef_rot_distance, layer_idx, node_idx
    ):
        # identification
        self.layer_idx = layer_idx
        self.node_idx = node_idx

        # graph search attributes
        self.cost = np.inf  # cumulative cost to reach node, used for Dijkstra's
        self.prev = (
            None  # previous node on best path, used for backtracking after Dijkstra's
        )

        # metrics
        self.q = q  # joint configuration (numpy array)
        self.target_pos = target_pos  # target end-effector position (numpy array)
        self.eef_pos_distance = (
            eef_pos_distance  # end-effector position distance from target
        )
        self.eef_rot_distance = (
            eef_rot_distance  # end-effector rotation distance from target
        )
        self.manipulability = 0.0  # manipulability score at self.q configuration

        # flags for debugging
        self.is_in_self_collision = False
        self.is_within_joint_limits = False
        # self.is_analytical_solution = False


class SphereScorer:
    def __init__(self, station, kinematics_solver):
        # Weights for cost function
        self.w_joint_dist = 1.0
        self.w_eef_pos_dist = 1.0  # avg. pos dist. is roughly 0.051086 m
        self.w_eef_rot_dist = 1.0  # avg. rot dist. is roughly 0.000139 rad
        self.w_manipulability = -0.05  # manipulability score ranges from 0 to 1

        self.kinematics_solver = kinematics_solver

        # All Drake station components
        self.station = station
        self.internal_station = self.station.internal_station
        # self.internal_plant = station.get_internal_plant()
        # self.plant_context = station.get_internal_plant_context()
        # self.internal_sg = self.internal_plant.get_scene_graph()

        # internal_station_diagram_context = self.station.internal_station.CreateDefaultContext()
        # scene_graph_context = self.internal_sg.GetMyContextFromRoot(internal_station_diagram_context)

        self.optimization_plant = self.internal_station.get_optimization_plant()
        self.optimization_plant_context = (
            self.internal_station.get_optimization_plant_context()
        )
        self.optimization_diagram = self.internal_station.get_optimization_diagram()
        self.optimization_diagram_context = (
            self.internal_station.get_optimization_diagram_context()
        )
        self.optimization_diagram_sg = (
            self.internal_station.get_optimization_diagram_sg()
        )
        self.optimization_diagram_sg_context = (
            self.internal_station.get_optimization_diagram_sg_context()
        )

        # Internal station
        # self.internal_diagram_context = self.station.internal_station.CreateDefaultContext()
        # self.internal_plant_updater = self.station.internal_station.GetSubsystemByName("plant_updater")
        # self.updater_context = self.station.internal_station.get_plant_context()

        # Frame
        self.tip_frame = self.optimization_plant.GetFrameByName("microscope_tip_link")

        # Joint limits
        self.joint_lower_limits = self.optimization_plant.GetPositionLowerLimits()
        self.joint_upper_limits = self.optimization_plant.GetPositionUpperLimits()

    # ===================================================================
    # Cost function components
    # ===================================================================
    def edge_cost(self, prev_node, curr_node):
        """
        Cost of moving from prev_node to curr_node.
        1. Joint distance: Distance between two joint configurations. We want to minimize large joint movements.
        2. End-effector position distance: Curr node's end-effector distance from desired pose.
        3. End-effector rotation distance: Curr node's end-effector rotation distance from desired pose.

        Args:
            prev_node (Node): Starting node.
            curr_node (Node): Ending node.

        Returns:
            cost (float): Cost of moving from prev_node to curr_node.
        """

        cost = (
            self.w_joint_dist * self.joint_distance(prev_node, curr_node)
            + self.w_eef_pos_dist * curr_node.eef_pos_distance
            + self.w_eef_rot_dist * curr_node.eef_rot_distance
            # + self.w_manipulability * curr_node.manipulability
        )

        return cost

    def eef_distances(self, target_pose, current_pose):
        """
        Compute distance of end-effector from desired pose (position + orientation).

        Args:
            target_pose (np.array): Target pose (position xyz + quaternion xyzw) [7,].
            current_pose (np.array): Current end-effector pose (position xyz + quaternion xyzw) [7,].

        Returns:
            distance (float): Combined position and orientation distance.
        """

        # =========================== TODO: Check this function, I just used previous copilot without verifying ===========================

        # Position distance
        eef_distance_pos = current_pose[0:3] - target_pose[0:3]
        eef_distance_pos = np.linalg.norm(eef_distance_pos)

        # Orientation distance (both quaternions should be in xyzw format)
        target_quat = target_pose[3:]  # xyzw format
        target_rot = Rotation.from_quat(target_quat)
        current_quat = current_pose[3:]  # xyzw format
        current_rot = Rotation.from_quat(current_quat)

        # Compute geodesic distance on SO(3)
        R_error = current_rot.as_matrix().T @ target_rot.as_matrix()
        trace = np.trace(R_error)

        # Clip to avoid numerical issues with arccos
        eef_angle_dist = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))

        return eef_distance_pos, eef_angle_dist

    def joint_distance(self, node1, node2):
        return np.linalg.norm(node2.q - node1.q)

    def is_within_joint_limits(self, q):
        # print("Joint limits check:")
        # print(" lower limits:", self.joint_lower_limits)
        # print(" upper limits:", self.joint_upper_limits)
        # print(" current q:", q)
        is_within = np.all(q >= self.joint_lower_limits) and np.all(
            q <= self.joint_upper_limits
        )
        # print(" within limits:", is_within)
        # # if not, print which joint(s) are not within limits
        # if not is_within:
        #     for i in range(len(q)):
        #         if q[i] < self.joint_lower_limits[i]:
        #             print(f" Joint {i+1} below lower limit: {q[i]:.4f} < {self.joint_lower_limits[i]:.4f}")
        #         if q[i] > self.joint_upper_limits[i]:
        #             print(f" Joint {i+1} above upper limit: {q[i]:.4f} > {self.joint_upper_limits[i]:.4f}")
        # # input()
        return is_within

    def is_in_self_collision(self, q):
        # 1) Set context to q
        self.optimization_plant.SetPositions(self.optimization_plant_context, q)
        # 2) Call scene graph to check for collisions given this new context
        query_object = self.optimization_diagram_sg.get_query_output_port().Eval(
            self.optimization_diagram_sg_context
        )
        has_collision = query_object.HasCollisions()
        # 3) Visualize for sanity check

        self.optimization_diagram.ForcedPublish(self.optimization_diagram_context)
        # print(f"Visualizing q: {q}")
        # print(f"Collision: {has_collision}")
        # time.sleep(0.01)  # Pause to let you see it

        # print("Is there a collision?", has_collision)
        return has_collision

    def djikstra_search(self, layers):
        # Initialize start nodes (first layer)
        for node in layers[0]:
            node.cost = 0

        # Process layers
        for layer_idx in range(1, len(layers)):
            current_layer = layers[layer_idx]
            previous_layer = layers[layer_idx - 1]

            for curr_node in current_layer:
                for prev_node in previous_layer:
                    cost = prev_node.cost + self.edge_cost(prev_node, curr_node)
                    if cost < curr_node.cost:
                        curr_node.cost = cost
                        curr_node.prev = prev_node

        # Backtrack to find best path
        end_layer = layers[-1]
        best_end_node = min(end_layer, key=lambda node: node.cost)

        path = []
        current_node = best_end_node
        while current_node is not None:
            path.append(current_node)
            current_node = current_node.prev
        path.reverse()

        return path, best_end_node.cost

    def generate_graph(self, waypoints, num_elbow_angles=1):
        """
        Generate graph of IK solutions for hemisphere scanning. This graph is traversed with Dijksra's
        later to find the optimal path.

        Args:
            waypoints (list): List of waypoints, each waypoint is a list of target poses.
            num_elbow_angles (int): Number of elbow angles to sample for each waypoint.
        """
        layers = []
        max_manipulability = -np.inf

        # Running averages for eef distances
        total_eef_pos_dist = 0.0
        total_eef_rot_dist = 0.0
        num_configs = 0

        for layer_idx, waypoint_set in enumerate(waypoints):
            layer_nodes = []

            for node_idx, target_pose in enumerate(waypoint_set):
                target_pos = target_pose[0:3]
                target_quat = target_pose[3:]
                target_rotmat = Rotation.from_quat(target_quat).as_matrix()

                # Compute IK for multiple elbow angles
                elbow_angles = np.linspace(
                    0, 2 * np.pi, num_elbow_angles, endpoint=False
                )
                ik_sols = []
                for idx in range(num_elbow_angles):
                    sols = self.kinematics_solver.IK_for_microscope(
                        target_rotmat,
                        target_pos,
                        psi=elbow_angles[idx],
                    )
                    ik_sols.append(sols)

                ik_sols = np.vstack(ik_sols)

                for sol_idx in range(ik_sols.shape[0]):
                    q_sol = ik_sols[sol_idx, :]

                    # Use forward kinematics to get end-effector pose given q_sol
                    self.optimization_plant.SetPositions(
                        self.optimization_plant_context, q_sol
                    )

                    X_W_TIP = self.optimization_plant.CalcRelativeTransform(
                        self.optimization_plant_context,
                        self.optimization_plant.world_frame(),
                        self.tip_frame,
                    )

                    microscope_tip_pos = X_W_TIP.translation()  # numpy array [x, y, z]
                    # get microscope tip quat in xyzw format
                    microscope_tip_quat = Rotation.from_matrix(
                        X_W_TIP.rotation().matrix()
                    ).as_quat()  # xyzw format

                    microscope_tip_pose = np.concatenate(
                        (microscope_tip_pos, microscope_tip_quat)
                    )
                    eef_pos_dist, eef_rot_dist = self.eef_distances(
                        target_pose, microscope_tip_pose
                    )

                    # # Debug: Print first waypoint errors
                    # if layer_idx == 0 and node_idx == 0 and sol_idx == 0:
                    #     print(
                    #         colored(
                    #             "\n=== Debug: First waypoint IK/FK validation ===",
                    #             "yellow",
                    #         )
                    #     )
                    #     print(f"Target position: {target_pos}")
                    #     print(f"FK position:     {microscope_tip_pos}")
                    #     print(f"Position error:  {eef_pos_dist:.6f} m")
                    #     print(
                    #         f"Rotation error:  {eef_rot_dist:.6f} rad ({np.degrees(eef_rot_dist):.3f}°)"
                    #     )
                    #     print(
                    #         colored(
                    #             "==============================================\n",
                    #             "yellow",
                    #         )
                    #     )

                    node = Node(
                        q=q_sol,
                        target_pos=target_pos,
                        eef_pos_distance=eef_pos_dist,
                        eef_rot_distance=eef_rot_dist,
                        layer_idx=layer_idx,
                        node_idx=node_idx,
                    )

                    # Check joint limits and self-collision
                    node.is_within_joint_limits = self.is_within_joint_limits(q_sol)
                    node.is_in_self_collision = self.is_in_self_collision(
                        q_sol
                    )  # TODO: Integrate with above, since it already runs SetPositions()

                    # Add manipulability score
                    # node.manipulability = self.manipulability_score(q_sol)
                    # if node.manipulability > max_manipulability:
                    #     max_manipulability = node.manipulability

                    # Update running averages
                    total_eef_pos_dist += eef_pos_dist
                    total_eef_rot_dist += eef_rot_dist
                    num_configs += 1

                    layer_nodes.append(node)

            # Filter out nodes as needed
            total_sols = len(layer_nodes)
            num_self_collision_sols = np.sum(
                [1 for node in layer_nodes if node.is_in_self_collision]
            )
            num_invalid_joint_limit_sols = np.sum(
                [1 for node in layer_nodes if not node.is_within_joint_limits]
            )
            # Filter out joint limit violations
            layer_nodes = [node for node in layer_nodes if node.is_within_joint_limits]
            # Filter out self-collisions
            layer_nodes = [
                node for node in layer_nodes if not node.is_in_self_collision
            ]

            # Cool visualization of total solutions vs. valid solutions
            print(colored("Layer " + str(layer_idx) + " solutions breakdown:", "cyan"))
            print(f"  Total IK solutions:                 {total_sols}")
            print(f"  Self-collision solutions:          {num_self_collision_sols}")
            print(
                f"  Invalid joint limit solutions:     {num_invalid_joint_limit_sols}"
            )
            print(f"                                    _____")
            print(f"  Valid solutions remaining:         ={len(layer_nodes)}\n")
            print(
                "" + colored("==============================================\n", "cyan")
            )

            # Filter out invalid solutions
            # layer_nodes = [node for node in layer_nodes if node.is_analytical_solution]

            # Add big score for any layers with no valid IK solutions
            if len(layer_nodes) == 0:
                print(
                    colored(
                        "No valid IK solutions found for layer "
                        + str(layer_idx)
                        + ". Returning with high cost",
                        "red",
                    )
                )
                return [], -1, -1

            layers.append(layer_nodes)

        # Compute averages
        avg_eef_pos_dist = total_eef_pos_dist / num_configs if num_configs > 0 else 0.0
        avg_eef_rot_dist = total_eef_rot_dist / num_configs if num_configs > 0 else 0.0

        return layers, avg_eef_pos_dist, avg_eef_rot_dist


def generate_hemisphere_waypoints(center, radius, num_points, num_rotations):
    """
    Generate evenly distributed waypoints on a hemisphere using golden angle sampling.

    This method uses the Fibonacci/golden angle spiral to distribute points uniformly
    on the surface of a hemisphere (upper half of a sphere).

    Args:
        center (np.ndarray): Center of the hemisphere [x, y, z]
        radius (float): Radius of the hemisphere in meters
        num_points (int): Number of waypoints to generate
        num_rotations (int): Number of rotated coordinate frames at each waypoint (rotated around surface normal)

    Returns:
        np.ndarray: Array of shape (num_points, num_rotations, 7) where the last dimension
                    contains [x, y, z, qx, qy, qz, qw] for each waypoint
    """

    # Initialize 3D array: (num_points, num_rotations, 7)
    waypoints_array = np.zeros((num_points, num_rotations, 7))

    # Golden angle in radians
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ≈ 2.399963 radians ≈ 137.508°

    for i in range(num_points):
        # Normalized height (1 to 0 for hemisphere, starting from top)
        # Using sqrt for more uniform distribution
        z_normalized = 1.0 - (i / (num_points - 1)) if num_points > 1 else 1.0

        # Height on hemisphere (starts from radius, goes down to 0)
        z = radius * z_normalized

        # Radius at this height (circle radius decreases as we go up)
        radius_at_height = radius * np.sqrt(1 - z_normalized**2)

        # Angle using golden angle spiral
        theta = golden_angle * i

        # Convert to Cartesian coordinates
        x = radius_at_height * np.cos(theta)
        y = radius_at_height * np.sin(theta)

        point = np.array([x, y, z])

        # Compute coordinate frame at this point
        # Z-axis: normal to sphere surface (pointing outward from center)
        z_axis = (
            point / np.linalg.norm(point)
            if np.linalg.norm(point) > 1e-10
            else np.array([0, 0, 1])
        )

        # X-axis: projection of [1,0,0] onto tangent plane
        reference = np.array([1.0, 0.0, 0.0])

        # Project reference vector onto tangent plane: v_proj = v - (v·n)n
        x_axis = reference - np.dot(reference, z_axis) * z_axis
        x_axis_norm = np.linalg.norm(x_axis)

        # Handle singularity: when point is aligned with [1,0,0] or very close
        if x_axis_norm < 1e-6:
            # Use [0,1,0] as backup reference
            reference = np.array([0.0, 1.0, 0.0])
            x_axis = reference - np.dot(reference, z_axis) * z_axis
            x_axis_norm = np.linalg.norm(x_axis)

        x_axis = x_axis / x_axis_norm

        # Y-axis: cross product to complete right-handed frame
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)  # Normalize for numerical stability

        # Build rotation matrix [x_axis, y_axis, z_axis] as columns
        base_rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # Generate num_rotations coordinate frames by rotating around z_axis (surface normal)
        for rot_idx in range(num_rotations):
            # Compute rotation angle (evenly distributed)
            rotation_angle = (
                (2 * np.pi * rot_idx) / num_rotations if num_rotations > 1 else 0.0
            )

            # Create rotation matrix around z-axis
            # R_z(θ) rotates the x and y axes while keeping z fixed
            cos_angle = np.cos(rotation_angle)
            sin_angle = np.sin(rotation_angle)
            R_z = np.array(
                [[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]]
            )

            # Apply rotation in the local frame: R_rotated = R_base * R_z
            rotation_matrix = base_rotation_matrix @ R_z

            # Convert to quaternion [qx, qy, qz, qw]
            rot = Rotation.from_matrix(rotation_matrix)
            quat = rot.as_quat()  # Returns [qx, qy, qz, qw]

            # Store in 3D array: [x, y, z, qx, qy, qz, qw]
            point_global = point + center
            waypoints_array[i, rot_idx, :3] = point_global
            waypoints_array[i, rot_idx, 3:] = quat

    # ===================================================================
    # Rotate hemisphere to bulge toward -x instead of +z
    # ===================================================================
    # Rotation matrix: 90 degrees around y-axis to transform +z → -x
    R_y_90 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    for i in range(waypoints_array.shape[0]):
        for j in range(waypoints_array.shape[1]):
            # Rotate position (relative to center)
            pos_relative = waypoints_array[i, j, :3] - center
            pos_rotated = R_y_90 @ pos_relative
            waypoints_array[i, j, :3] = pos_rotated + center

            # Rotate orientation quaternion
            quat = waypoints_array[i, j, 3:]
            rot_original = Rotation.from_quat(quat)
            rot_transform = Rotation.from_matrix(R_y_90)
            rot_new = rot_transform * rot_original
            waypoints_array[i, j, 3:] = rot_new.as_quat()

    return waypoints_array


def plot_hemisphere_waypoints(
    center, waypoints_array, radius, save_path, save_plot=True
):
    """
    Plot waypoints with coordinate frames on a 3D hemisphere and optionally save the figure.

    Args:
        center (np.ndarray): Center of the hemisphere [x, y, z]
        waypoints_array (np.ndarray): Array of shape (num_points, num_rotations, 7) containing
                                       [x, y, z, qx, qy, qz, qw] for each waypoint
        radius (float): Radius of the hemisphere
        save_path (str or Path, optional): Path to save the plot image
        save_plot (bool): Whether to save the plot to file
    """
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Extract positions for plotting (flatten first two dimensions)
    positions = waypoints_array[:, :, :3].reshape(-1, 3)

    # Plot waypoints
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c="red",
        marker="o",
        s=30,
        alpha=0.6,
        label="Waypoints",
    )

    # Draw hemisphere surface (wireframe)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi / 2, 20)  # Only upper hemisphere
    x_surf = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y_surf = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z_surf = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.15, color="lightblue")

    # Draw base circle
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    x_circle = radius * np.cos(theta_circle) + center[0]
    y_circle = radius * np.sin(theta_circle) + center[1]
    z_circle = np.zeros_like(theta_circle) + center[2]
    ax.plot(
        x_circle, y_circle, z_circle, "b--", linewidth=2, alpha=0.4, label="Base circle"
    )

    # Plot coordinate frames at waypoints
    # For each unique position: show z-axis in blue (once) and all x-axes in red (one per rotation)
    frame_scale = radius * 0.15  # Scale for coordinate frame axes

    # Iterate through points (first dimension)
    for i in range(waypoints_array.shape[0]):
        # Get position (same for all rotations at this point)
        point = waypoints_array[i, 0, :3]

        # Plot z-axis once (it's the same for all rotations at this point)
        first_quat = waypoints_array[i, 0, 3:]
        rot = Rotation.from_quat(first_quat)
        rotation_matrix = rot.as_matrix()
        z_axis = rotation_matrix[:, 2]

        ax.quiver(
            point[0],
            point[1],
            point[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="blue",
            length=frame_scale,
            arrow_length_ratio=0.3,
            linewidth=2,
            alpha=0.8,
        )

        # Plot x-axis for each rotation at this point
        for j in range(waypoints_array.shape[1]):
            quat = waypoints_array[i, j, 3:]
            rot = Rotation.from_quat(quat)
            rotation_matrix = rot.as_matrix()
            x_axis = rotation_matrix[:, 0]

            ax.quiver(
                point[0],
                point[1],
                point[2],
                x_axis[0],
                x_axis[1],
                x_axis[2],
                color="red",
                length=frame_scale,
                arrow_length_ratio=0.3,
                linewidth=1.5,
                alpha=0.8,
            )

    # Add legend for coordinate frame axes
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="red",
            marker="o",
            linestyle="None",
            markersize=6,
            label="Waypoints",
        ),
        Line2D([0], [0], color="b", linestyle="--", linewidth=2, label="Base circle"),
        Line2D([0], [0], color="red", linewidth=2, label="X-axes (rotations)"),
        Line2D([0], [0], color="blue", linewidth=2, label="Z-axis (normal)"),
    ]

    # Labels and title
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_zlabel("Z (m)", fontsize=12)
    ax.set_title(
        f"Hemisphere Waypoints with Coordinate Frames\n"
        f"Golden Angle Sampling - Radius: {radius}m, Points: {len(positions)}",
        fontsize=14,
        fontweight="bold",
    )

    # Equal aspect ratio
    max_range = radius * 1.2  # Add margin for frames
    ax.set_xlim([-max_range + center[0], max_range + center[0]])
    ax.set_ylim([-max_range + center[1], max_range + center[1]])
    ax.set_zlim([center[2], max_range + center[2]])

    # Set aspect ratio accounting for z being half the range of x and y
    # x and y span 2*radius, z spans radius, so use [1, 1, 0.5]
    ax.set_box_aspect([1, 1, 0.5])

    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    # Save figure if path provided
    if save_plot:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path.absolute()}")

    plt.show()


def generate_hemisphere_joint_poses(
    station: IiwaHardwareStationDiagram,  # Station from hardware_station.py
    center: np.ndarray,
    radius: float,
    num_poses: int,
    num_rotations_per_pose: int,
    num_elbow_positions: int,
    kinematics_solver=None,
):
    """
    Generate joint poses for scanning hemisphere, while optimizing these parameters:
    """

    # ===================================================================
    # Params
    # ===================================================================
    save_plot = False
    if kinematics_solver is None:
        kinematics_solver = KinematicsSolver()
    sphere_scorer = SphereScorer(station, kinematics_solver)
    # ===================================================================
    # Generate all waypoints on hemisphere
    # ===================================================================
    waypoints = generate_hemisphere_waypoints(
        center, radius, num_poses, num_rotations_per_pose
    )

    if save_plot:
        print("Generating plot...")
        output_plot = (
            Path(__file__).parent.parent / "outputs" / "hemisphere_waypoints.png"
        )
        plot_hemisphere_waypoints(
            center, waypoints, radius, output_plot, save_plot=save_plot
        )
        print("Plot generation complete.")
        quit()

    # ===================================================================
    # Create graph for evaluating least-cost path
    # ===================================================================
    layers, avg_eef_pos_dist, avg_eef_rot_dist = sphere_scorer.generate_graph(
        waypoints,
        num_elbow_angles=num_elbow_positions,
    )

    # ===================================================================
    # Find least-cost path through graph
    # ===================================================================
    if layers == []:
        return np.inf, []

    # Step 3: Compute costs and find optimal path (Dijkstra's or A* algorithm)
    path, cost = sphere_scorer.djikstra_search(layers)

    # Print running averages
    print(colored("\n=== Running Averages Across All Configurations ===", "magenta"))
    print(f"Average EEF Position Distance: {avg_eef_pos_dist:.6f} m")
    print(
        f"Average EEF Rotation Distance: {avg_eef_rot_dist:.6f} rad ({np.degrees(avg_eef_rot_dist):.3f}°)"
    )
    print(colored("=================================================\n", "magenta"))

    return cost, path


def find_best_hemisphere_center(
    station: IiwaHardwareStationDiagram,
    hemisphere_centers: list,
    radius: float,
    num_poses: int,
    num_rotations_per_pose: int,
    num_elbow_positions: int,
    kinematics_solver=None,
):
    """
    Evaluate multiple hemisphere centers and return the one with the lowest cost path.

    Args:
        station (IiwaHardwareStationDiagram): Station from hardware_station.py
        hemisphere_centers (list): List of hemisphere center positions, each as np.ndarray [x, y, z]
        radius (float): Radius of the hemisphere
        num_poses (int): Number of waypoints on hemisphere
        num_rotations_per_pose (int): Number of rotations per waypoint
        num_elbow_positions (int): Number of elbow angles to sample
        kinematics_solver (KinematicsSolver, optional): Kinematics solver instance

    Returns:
        tuple: (best_center, best_cost, best_path)
            - best_center (np.ndarray): The hemisphere center with lowest cost
            - best_cost (float): The cost of the best path
            - best_path (list): The path nodes for the best solution
    """
    if not hemisphere_centers:
        raise ValueError("hemisphere_centers list cannot be empty")

    best_center = None
    best_cost = np.inf
    best_path = []
    results = []  # Store results for CSV

    print(colored(f"\n{'='*60}", "cyan"))
    print(
        colored(f"Evaluating {len(hemisphere_centers)} hemisphere centers...", "cyan")
    )
    print(colored(f"{'='*60}\n", "cyan"))

    for idx, center in enumerate(hemisphere_centers):
        print(
            colored(
                f"\n--- Testing Center {idx + 1}/{len(hemisphere_centers)}: {center} ---",
                "yellow",
            )
        )

        cost, path = generate_hemisphere_joint_poses(
            station=station,
            center=center,
            radius=radius,
            num_poses=num_poses,
            num_rotations_per_pose=num_rotations_per_pose,
            num_elbow_positions=num_elbow_positions,
            kinematics_solver=kinematics_solver,
        )

        print(colored(f"Center {idx + 1} cost: {cost:.6f}", "yellow"))

        # Store results
        results.append(
            {
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "cost": cost,
                "num_nodes": len(path) if path else 0,
            }
        )

        if cost < best_cost:
            best_cost = cost
            best_center = center
            best_path = path
            print(colored(f"✓ New best center found!", "green"))

    # Save results to CSV
    outputs_dir = Path(__file__).parent.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = outputs_dir / "hemisphere_center_costs.csv"

    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["center_x", "center_y", "center_z", "cost", "num_nodes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(colored(f"\nResults saved to: {csv_path}", "blue"))

    print(colored(f"\n{'='*60}", "cyan"))
    print(colored(f"Best hemisphere center: {best_center}", "green"))
    print(colored(f"Best cost: {best_cost:.6f}", "green"))
    print(colored(f"Path length: {len(best_path)} nodes", "green"))
    print(colored(f"{'='*60}\n", "cyan"))

    return best_center, best_cost, best_path
    # ===================================================================
    # Debugging
    # ===================================================================

    # # Test if IK solver can solve many values for first waypoint
    # print("Testing IK solver for first waypoint...")

    # # Get necessary values
    # internal_plant = station.get_internal_plant()
    # internal_sg = station.internal_station.get_scene_graph()
    # context = station.internal_station.CreateDefaultContext()
    # Test current robot location for self-collisions
