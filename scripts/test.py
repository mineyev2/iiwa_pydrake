import argparse

import matplotlib.pyplot as plt
import numpy as np

from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario

from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    JointSliders,
    MeshcatVisualizer,
    PiecewisePolynomial,
    Simulator,
    RigidTransform,
    Sphere,
    CoulombFriction,
    Rgba,
    InverseKinematics,
    SpatialInertia,
    UnitInertia,
    SceneGraphCollisionChecker,
    MinimumDistanceLowerBoundConstraint,
    KinematicTrajectoryOptimization,
    Solve,
    Meshcat,
    Rgba,
    RigidTransform,
    BsplineBasis,
    BsplineTrajectory,
    KnotVectorType
)

from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.primitives import FirstOrderLowPassFilter

from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra


def draw_sphere(meshcat, name, position, radius=0.01):

    rgba = Rgba(0.0, 1.0, 0.1, 0.5)

    meshcat.SetObject(
        name,
        Sphere(radius),
        rgba,
    )
    meshcat.SetTransform(
        name,
        RigidTransform(np.array(position)),
    )

def linear_bspline_for_trajopt(q_start, q_goal, num_control_points, duration=1.0, spline_order=3):
    """
    Creates a straight-line B-spline trajectory with exactly num_control_points
    to match trajopt.
    """
    q_start = np.asarray(q_start)
    q_goal = np.asarray(q_goal)
    num_joints = q_start.shape[0]
    
    # Create linear interpolation at each control point
    t_points = np.linspace(0, 1, num_control_points)
    control_points = np.outer(1 - t_points, q_start) + np.outer(t_points, q_goal)  # shape: (num_control_points, num_joints)
    control_points = control_points.T  # shape: (num_joints, num_control_points)
    
    # Make Bspline basis
    from pydrake.math import BsplineBasis, KnotVectorType
    basis = BsplineBasis(
        order=spline_order,
        num_basis_functions=num_control_points,
        type=KnotVectorType.kClampedUniform,
        initial_parameter_value=0.0,
        final_parameter_value=duration
    )
    
    return BsplineTrajectory(basis, control_points.tolist())

def main(use_hardware: bool, has_wsg: bool) -> None:
    scenario_data = (
    """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14.dmd.yaml
    - add_model:
        name: sphere_obstacle
        file: package://iiwa_setup/sphere_obstacle.sdf
    - add_weld:
        parent: world
        child: sphere_obstacle::sphere_body
        X_PC:
            translation: [0.5, 0.1, 0.7]
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
    )
    
    builder = DiagramBuilder()

    scenario = LoadScenario(data=scenario_data)
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario, has_wsg=has_wsg, use_hardware=use_hardware
        ),
    )

    # Set up teleop widgets
    controller_plant = station.get_iiwa_controller_plant()
    teleop = builder.AddSystem(
        JointSliders(
            station.internal_meshcat,
            controller_plant,
        )
    )

    builder.Connect(
        teleop.get_output_port(), station.GetInputPort("iiwa.position"),
    )

    if has_wsg:
        wsg_teleop = builder.AddSystem(WsgButton(station.internal_meshcat))
        builder.Connect(
            wsg_teleop.get_output_port(0), station.GetInputPort("wsg.position")
        )

    # Required for visualizing the internal station
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    ApplySimulatorConfig(scenario.simulator_config, simulator)
    simulator.set_target_realtime_rate(1.0)

    station.internal_meshcat.AddButton("Stop Simulation")
    station.internal_meshcat.AddButton("Move to Goal")


    # ====================================================================
    # Trajectory Optimization + TOPPRA Loop
    # ====================================================================

    # Define goal position and limits
    q_goal = np.array([0, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0])
    vel_limits = np.full(7, 0.5)  # rad/s
    acc_limits = np.full(7, 0.5)  # rad/s^2
    
    # Get optimization plant and context
    optimization_plant = station.internal_station._optimization_plant
    optimization_plant_context = station.internal_station._optimization_plant_context

    # Set up trajectory optimization
    num_control_points = 20
    trajopt = KinematicTrajectoryOptimization(
        num_positions=optimization_plant.num_positions(),
        num_control_points=num_control_points,
        # spline_order=2,
    )
    prog = trajopt.get_mutable_prog()

    # Add collision constraints
    min_distance = 0.01  # meters
    for i in range(num_control_points):
        qvars = trajopt.control_points()[:, i]
        print("Current qvars:", qvars)

        constraint = MinimumDistanceLowerBoundConstraint(
            plant=optimization_plant,
            bound=min_distance,
            plant_context=optimization_plant_context,
        )

        prog.AddConstraint(constraint, qvars)
    

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    move_clicks = -1
    trajectory = None
    trajectory_start_time = 0.0
    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        # Check if "Move to Goal" button was clicked
        new_move_clicks = station.internal_meshcat.GetButtonClicks("Move to Goal")
        if new_move_clicks > move_clicks:
            move_clicks = new_move_clicks
            print("Planning trajectory to goal...")

            # 1. Get current position
            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(station_context)
            
            # 2. Add extra constraints and costs
            linear_bspline = linear_bspline_for_trajopt(q_current, q_goal, duration=1.0, spline_order=3, num_control_points=num_control_points)
            trajopt.SetInitialGuess(linear_bspline)
            
            # Position
            trajopt.AddPathPositionConstraint(q_current, q_current, 0.0)
            trajopt.AddPathPositionConstraint(q_goal, q_goal, 1.0)
            # Velocity (TOPPRA assumes zero start and end velocities)
            trajopt.AddPathVelocityConstraint(
                np.zeros_like(q_current), np.zeros_like(q_current), 0.0
            )
            trajopt.AddPathVelocityConstraint(
                np.zeros_like(q_goal), np.zeros_like(q_goal), 1.0
            )
            # Costs
            trajopt.AddPathLengthCost(1.0)
            # trajopt.AddDurationCost(1.0)

            # 3. Solve optimization
            result = Solve(prog)

            internal_plant = station.get_internal_plant()
            internal_context = station.get_internal_plant_context()

            for i in range(num_control_points):
                q_sol = result.GetSolution(trajopt.control_points()[:, i])

                # TODO: Draw robot points along path for debugging
                internal_plant.SetPositions(internal_context, q_sol)
                X_WB = internal_plant.EvalBodyPoseInWorld(
                    internal_context,
                    internal_plant.GetBodyByName("iiwa_link_7"),
                )
                sphere_name = f"traj_point_{i}"
                draw_sphere(
                    station.internal_meshcat,
                    sphere_name,
                    X_WB.translation(),
                )

            if not result.is_success():
                print("Optimization failed!")
                continue

            print("Trajectory optimization succeeded!")
            geometric_path = trajopt.ReconstructTrajectory(result)

            # 4. Save path as img through matplotlib
            ts = np.linspace(geometric_path.start_time(), geometric_path.end_time(), 100)
            qs = np.array([geometric_path.value(t) for t in ts])
            plt.figure()
            for i in range(qs.shape[1]):
                plt.plot(ts, qs[:, i], label=f"Joint {i+1}")
            plt.xlabel("Time [s]")
            plt.ylabel("Joint Position [rad]")
            plt.title("Geometric Path Joint Positions")
            plt.legend()
            plt.savefig("geometric_path.png")
            plt.close()

            # 5. Reparameterize with TOPPRA
            trajectory = reparameterize_with_toppra(
                geometric_path,
                controller_plant, # NOTE: don't need optimization_plant since collision avoidance is solved
                velocity_limits=vel_limits,
                acceleration_limits=acc_limits,
            )

            # 6. Save reparameterized trajectory as img through matplotlib
            ts = np.linspace(trajectory.start_time(), trajectory.end_time(), 100)
            qs = np.array([trajectory.value(t) for t in ts])
            plt.figure()
            for i in range(qs.shape[1]):
                plt.plot(ts, qs[:, i], label=f"Joint {i+1}")
            plt.xlabel("Time [s]")
            plt.ylabel("Joint Position [rad]")
            plt.title("Reparameterized Trajectory Joint Positions")
            plt.legend()
            plt.savefig("reparameterized_trajectory.png")
            plt.close()

            print(f"✓ TOPPRA succeeded! Trajectory duration: {trajectory.end_time():.2f}s")
            trajectory_start_time = simulator.get_context().get_time()

        # If we have a trajectory, execute it
        if trajectory is not None:
            current_time = simulator.get_context().get_time()
            traj_time = current_time - trajectory_start_time
            
            if traj_time <= trajectory.end_time():
                q_desired = trajectory.value(traj_time)
                station_context = station.GetMyMutableContextFromRoot(
                    simulator.get_mutable_context()
                )
                station.GetInputPort("iiwa.position").FixValue(station_context, q_desired)
            else:
                print("✓ Trajectory execution complete!")
                trajectory = None
        
        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Move to Goal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )
    parser.add_argument(
        "--has_wsg",
        action="store_true",
        help="Whether the iiwa has a WSG gripper or not.",
    )
    args = parser.parse_args()
    main(use_hardware=args.use_hardware, has_wsg=args.has_wsg)
