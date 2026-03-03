import numpy as np

from utils.geometric_subproblems import rot, sp_1, sp_2, sp_3
from utils.sew_stereo import SEWStereo

"""
Geometry-based kinematics for KUKA LBR iiwa 14 R820 robot.

Taken from https://github.com/kczttm/SEW-Geometric-Teleop

Author: Roman Mineyev
"""

"""
Naming conventions:
- Frames:
    - 0: Base frame (in our case the world frame)
    - 7: End-effector frame (link 7)
    - M: Microscope mount frame (attached to end-effector)
    - S: Shoulder (joint 1 position)
    - E: Elbow (joint 4 position)
    - W: Wrist (joint 7 position)
- Variables:
    - R_AB: Rotation matrix from frame b to frame a
    - p_AB_C: Position vector from point A to point B expressed in frame C
    - q_i: Joint angle for joint i

vector from a to b expressed in frame c:


"""


class KinematicsSolver:
    """
    Kinematics solver for KUKA LBR iiwa 14 R820 robot.
    """

    def __init__(self, station):
        self.kin = self.get_kin()  # Get kinematic parameters of KUKA iiwa 14 R820 robot
        self.R_7M, self.p_7M_in_7 = self.get_microscope_offsets(
            station
        )  # Get the microscope mount offsets

        # self.r, self.v = np.array([-1, 0, 0]), np.array([0, 1, 0])
        self.r, self.v = np.array([0, 0, -1]), np.array([0, 1, 0])

    def get_microscope_offsets(self, station):
        """
        Calculate the microscope mount offsets based on the station's plant.

        NOTE: We use joint 6 position in SDF = joint 7 position in URF convention.
        This is just how the IK is set up.
        """

        # Get plant and context
        plant = station.get_internal_plant()
        context = station.get_internal_plant_context()
        joint6_frame = plant.GetFrameByName("iiwa_link_6")
        tip_frame = plant.GetFrameByName("microscope_tip_link")
        link7_frame = plant.GetFrameByName("iiwa_link_7")

        # Get relevant transforms
        p_07 = plant.CalcRelativeTransform(
            context, plant.world_frame(), joint6_frame
        ).translation()  # joint 7 pos in URDF/get_kin() = joint 6 pos in SDF
        p_0M = plant.CalcRelativeTransform(
            context, plant.world_frame(), tip_frame
        ).translation()
        R_70 = (
            plant.CalcRelativeTransform(context, plant.world_frame(), link7_frame)
            .inverse()
            .rotation()
        )  # joint 7 rot in URF/get_kin() = joint 7 rot in SDF

        p_7M = p_0M - p_07
        p_7M_in_7 = R_70.multiply(
            p_7M
        )  # Vector from joint 6 to microscope tip in joint 7 frame. Called p_M7 because joint 6 is link 7 in get_kin()

        # Debugging prints
        # print("Microscope tip position in world:", p_0M)
        # print("Joint 7 position in world:", p_07)
        # print("Microscope mount offset p_7M:", p_7M)
        # print("Microscope mount offset p_7M_in_7:", p_7M_in_7)
        # Microscope offsets (NOTE: I just know these so I didn't bother calculating. Assuming that it won't change)
        R_7M = np.array(
            [
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1],
            ]
        )

        return R_7M, p_7M_in_7

    def get_kin(self):
        """
        Get the kinematic parameters of the KUKA LBR iiwa 14 R820 robot.
        """

        # LBR iiwa 14 R820
        ey = np.array([0, 1, 0])
        ez = np.array([0, 0, 1])
        zv = np.zeros(3)

        # P: 3x7 matrix, each column is a vector
        kin_P = np.column_stack(
            [
                (0.1575 + 0.2025) * ez,
                zv,
                (0.2045 + 0.2155) * ez,
                zv,
                (0.1845 + 0.2155) * ez,
                zv,
                zv,
                (0.0810 + 0.0450) * ez,
            ]
        )

        # H: 3x7 matrix, each column is a joint axis
        kin_H = np.column_stack([ez, ey, ez, -ey, ez, ey, ez])

        # joint_type: 7-element array, all revolute (0)
        joint_type = np.zeros(7, dtype=int)

        return {"P": kin_P, "H": kin_H, "joint_type": joint_type}

    def IK_for_microscope_multiple_elbows(
        self, R_0M, p_0M, num_elbow_angles=50, track_elbow_angle=False
    ):
        """
        Solve the inverse kinematics for the KUKA LBR iiwa 14 R820 robot with a microscope mount, sampling multiple elbow configurations.

        Args:
            R_0M (np.ndarray): 3x3 rotation matrix from base to microscope mount.
            p_0M (np.ndarray): 3-element position vector of the microscope mount in the base frame.
            num_elbow_angles (int): Number of elbow angle samples to generate.
            track_elbow_angle (bool): If True, return elbow angles alongside solutions.

        Returns:
            Qs (np.ndarray): An Nx7 array of N IK solutions, where each row is a 7-element vector of joint angles.
            elbow_angles_list (list): (Optional) List of elbow angles corresponding to each solution in Qs. Only returned if track_elbow_angle=True.
        """

        Qs = []
        elbow_angles_list = []
        elbow_angles = np.linspace(-np.pi, np.pi, num_elbow_angles)
        for i in range(num_elbow_angles):
            Q = self.IK_for_microscope(R_0M, p_0M, psi=elbow_angles[i])
            if Q.shape[0] > 0:  # Only append if solutions were found
                Qs.append(Q)
                if track_elbow_angle:
                    # Add the elbow angle for each solution in Q
                    elbow_angles_list.extend([elbow_angles[i]] * Q.shape[0])

        Qs = np.vstack(Qs) if Qs else np.array([]).reshape(0, 7)

        if track_elbow_angle:
            return Qs, elbow_angles_list
        else:
            return Qs

    def IK_for_microscope(self, R_0M, p_0M, psi=None):
        """
        Solve the inverse kinematics for the KUKA LBR iiwa 14 R820 robot with a microscope mount.

        Args:
            R_0M (np.ndarray): 3x3 rotation matrix from base to microscope mount.
            p_0M (np.ndarray): 3-element position vector of the microscope mount in the base frame.
        """

        if psi is None:
            print("No psi angle provided, defaulting to 0 (fully extended elbow).")
            psi = 0  # Default psi angle if not provided

        # Adjust end-effector position to account for microscope mount offset
        R_07 = R_0M @ self.R_7M.T  # Rot from 7 to M to 0 => Rot from 0 to 7
        p_7M = R_07 @ self.p_7M_in_7
        p_07 = p_0M - p_7M

        # Solve IK using standard kuka_IK method

        sew_stereo = SEWStereo(self.r, self.v)

        # # Print what values we are solving for:
        # print("Solving IK for R_0M:\n", R_0M)
        # print("Solving IK for p_0M:", p_0M)
        # print("Adjusted R_07 for IK:\n", R_07)
        # print("Adjusted p_07 for IK:", p_07)
        # print("Using psi angle:", psi)

        return self.kuka_IK(R_07, p_07, sew_stereo, psi)

    def kuka_IK_for_multiple_elbows(self, R_07, p_07, num_elbow_angles=50):
        """
        Solve IK for multiple elbow configurations by sampling the SEW angle.
        Useful to later parse and find the closest solution to current joint angles.
        """

        Qs = []
        elbow_angles = np.linspace(-np.pi, np.pi, num_elbow_angles)

        sew_stereo = SEWStereo(self.r, self.v)
        for i in range(num_elbow_angles):
            Q = self.kuka_IK(R_07, p_07, sew_stereo, elbow_angles[i])
            Qs.append(Q)

        Qs = np.vstack(Qs) if Qs else np.array([]).reshape(0, 7)
        return Qs

    def find_closest_solution(self, Q, q, return_index=False):
        """
        Find the closest IK solution to the current joint configuration.
        NOTE: Assumes joint limits are between -180 and 180. Don't need to do wrapping in this case.

        Args:
            Q (np.ndarray): An Nx7 array of N IK solutions.
            q_current (np.ndarray): A 7-element vector of current joint angles.
            return_index (bool): If True, also return the index of the closest solution.

        Returns:
            q_closest (np.ndarray): A 7-element vector of the closest joint angles.
            closest_index (int): (Optional) Index of the closest solution. Only returned if return_index=True.
        """
        if Q.shape[0] == 0:
            raise ValueError("No IK solutions found.")

        # Calculate the distance from each solution to the current joint configuration
        distances = np.linalg.norm(Q - q, axis=1)

        # Find the index of the closest solution
        closest_index = np.argmin(distances)

        if return_index:
            return Q[closest_index], closest_index
        else:
            return Q[closest_index]

    def kuka_IK(self, R_07, p_07, sew_class, psi):
        """
        Solve the inverse kinematics for the KUKA LBR iiwa 14 R820 robot using geometric methods.

        Args:
            R_07 (np.ndarray): 3x3 rotation matrix from base to link 7.
            p_07 (np.ndarray): 3-element position vector of link 7 in the base frame.
            sew_class: An instance of the SEW class for solving the shoulder-elbow-wrist configuration.
            psi (float): The SEW angle.

        Returns:
            Q (np.ndarray): An Nx7 array of N IK solutions, where each row is a 7-element vector of joint angles.
        """

        kin = self.kin
        Q = []
        # is_LS_vec = []

        # Find wrist position
        # W = p_0T - R_07 @ kin["P"][:, 7]
        W = p_07

        # Find shoulder position
        S = kin["P"][:, 0]

        # Use subproblem 3 to find theta_SEW
        d_S_E = np.linalg.norm(np.sum(kin["P"][:, 1:4], axis=1))
        d_E_W = np.linalg.norm(np.sum(kin["P"][:, 4:7], axis=1))
        p_17 = W - S
        e_17 = p_17 / np.linalg.norm(p_17)

        # SEW inverse kinematics
        _, n_SEW = sew_class.inv_kin(S, W, psi)
        theta_SEW, theta_SEW_is_LS = sp_3(d_S_E * e_17, p_17, n_SEW, d_E_W)

        # Pick theta_SEW > 0 for correct half-plane
        q_SEW = np.max(theta_SEW)
        p_S_E = rot(n_SEW, q_SEW) @ (d_S_E * e_17)
        E = p_S_E + S

        # Find q1, q2 using subproblem 2
        h_1 = kin["H"][:, 0]
        h_2 = kin["H"][:, 1]
        p_S_E_0 = np.sum(kin["P"][:, 1:4], axis=1)
        t1, t2, t12_is_ls = sp_2(p_S_E, p_S_E_0, -h_1, h_2)

        for i_q12 in range(len(t1)):
            q1 = t1[i_q12]
            q2 = t2[i_q12]

            # Find q3, q4 using subproblem 2
            h_3 = kin["H"][:, 2]
            h_4 = kin["H"][:, 3]
            p_E_W_0 = np.sum(kin["P"][:, 4:7], axis=1)
            p_E_W = W - E

            R_2 = rot(h_1, q1) @ rot(h_2, q2)
            t3, t4, t34_is_ls = sp_2(R_2.T @ p_E_W, p_E_W_0, -h_3, h_4)

            for i_q34 in range(len(t3)):
                q3 = t3[i_q34]
                q4 = t4[i_q34]

                # Find q5, q6 using subproblem 2
                h_5 = kin["H"][:, 4]
                h_6 = kin["H"][:, 5]
                R_4 = R_2 @ rot(h_3, q3) @ rot(h_4, q4)
                t5, t6, t56_is_ls = sp_2(
                    R_4.T @ R_07 @ kin["H"][:, 6], kin["H"][:, 6], -h_5, h_6
                )

                for i_q56 in range(len(t5)):
                    q5 = t5[i_q56]
                    q6 = t6[i_q56]

                    # Find q7
                    h_7 = kin["H"][:, 6]
                    R_6 = R_4 @ rot(h_5, q5) @ rot(h_6, q6)
                    q7, q7_is_ls = sp_1(h_6, R_6.T @ R_07 @ h_6, h_7)

                    q_i = np.array([q1, q2, q3, q4, q5, q6, q7])
                    Q.append(q_i)
                    # overall_is_ls = (
                    #     theta_SEW_is_LS
                    #     or t12_is_ls
                    #     or t34_is_ls
                    #     or t56_is_ls
                    #     or q7_is_ls
                    # )
                    # is_LS_vec.append(overall_is_ls)

        # Q = np.column_stack(Q) if Q else np.array([]).reshape(7, 0)
        # return Q, is_LS_vec

        # NOTE: I just like it this way. Each sol is a row now.
        Q = np.vstack(Q) if Q else np.array([]).reshape(0, 7)
        return Q
