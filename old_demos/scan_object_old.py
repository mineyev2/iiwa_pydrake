# General imports
import argparse

from enum import Enum, auto
from pathlib import Path

import numpy as np

# Drake imports
from manipulation.meshcat_utils import WsgButton
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import LoadScenario
from pydrake.all import (
    ApplySimulatorConfig,
    DiagramBuilder,
    JointSliders,
    MeshcatVisualizer,
    Rgba,
    Simulator,
    Solve,
)
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.primitives import FirstOrderLowPassFilter
from termcolor import colored

# Personal files
from iiwa_setup.iiwa import IiwaForwardKinematics, IiwaHardwareStationDiagram
from iiwa_setup.motion_planning.toppra import reparameterize_with_toppra
from iiwa_setup.util.traj_planning import (
    add_collision_constraints_to_trajectory,
    resolve_with_toppra,
    setup_trajectory_optimization_from_q1_to_q2,
)
from iiwa_setup.util.visualizations import draw_sphere
from utils.hemisphere_solver import load_joint_poses_from_csv
from utils.kuka_geo_kin import KinematicsSolver


class State(Enum):
    IDLE = auto()
    INITIAL_GUESS_PLANNING = auto()
    FINAL_PLANNING = auto()
    MOVING = auto()
    ERROR = auto()


def main(use_hardware: bool) -> None:
    scenario_data = """
    directives:
    - add_directives:
        file: package://iiwa_setup/iiwa14_microscope.dmd.yaml
    # - add_model:
    #     name: sphere_obstacle
    #     file: package://iiwa_setup/sphere_obstacle.sdf
    # - add_weld:
    #     parent: world
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

    # ===================================================================
    # Diagram Setup
    # ===================================================================
    builder = DiagramBuilder()

    # Load scenario
    scenario = LoadScenario(data=scenario_data)
    hemisphere_pos = np.array([0.6666666, 0.0, 0.444444])
    hemisphere_radius = 0.05
    station: IiwaHardwareStationDiagram = builder.AddNamedSystem(
        "station",
        IiwaHardwareStationDiagram(
            scenario=scenario,
            hemisphere_pos=hemisphere_pos,
            hemisphere_radius=hemisphere_radius,
            use_hardware=use_hardware,
        ),
    )

    # Load all values I use later
    controller_plant = station.get_iiwa_controller_plant()

    # Load teleop sliders
    teleop = builder.AddSystem(
        JointSliders(
            station.internal_meshcat,
            controller_plant,
        )
    )

    # Add connections
    builder.Connect(
        teleop.get_output_port(),
        station.GetInputPort("iiwa.position"),
    )

    # Visualize internal station with Meshcat
    _ = MeshcatVisualizer.AddToBuilder(
        builder, station.GetOutputPort("query_object"), station.internal_meshcat
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
    station.internal_meshcat.AddButton("Plan Initial Guess Trajectory")
    station.internal_meshcat.AddButton("Plan Final Trajectory")
    station.internal_meshcat.AddButton("Move to Goal")

    # ====================================================================
    # Compute all joint poses for sphere scanning
    # ====================================================================
    # kinematics_solver = KinematicsSolver(station)
    # _, path_joint_poses = generate_hemisphere_joint_poses(
    #     station=station,
    #     center=hemisphere_pos,
    #     radius=hemisphere_radius,
    #     num_poses=30,
    #     num_rotations_per_pose=7,
    #     num_elbow_positions=10,
    #     kinematics_solver=kinematics_solver,
    # )

    # Load joint poses from CSV
    joint_poses_file = Path(__file__).parent.parent / "outputs" / "joint_poses.csv"
    path_joint_poses = load_joint_poses_from_csv(joint_poses_file)

    # ====================================================================
    # Main Simulation Loop
    # ====================================================================
    move_clicks = 0
    plan_initial_guess_clicks = 0
    final_plan_clicks = 0
    trajectory = None
    executing_trajectory = False
    trajectory_start_time = 0.0
    path_idx = 15
    vel_limits = np.full(7, 1.0)  # rad/s
    acc_limits = np.full(7, 1.0)  # rad/s^2

    state = State.INITIAL_GUESS_PLANNING

    while station.internal_meshcat.GetButtonClicks("Stop Simulation") < 1:
        new_move_clicks = station.internal_meshcat.GetButtonClicks("Move to Goal")
        new_plan_initial_guess_clicks = station.internal_meshcat.GetButtonClicks(
            "Plan Initial Guess Trajectory"
        )
        new_plan_final_clicks = station.internal_meshcat.GetButtonClicks(
            "Plan Final Trajectory"
        )

        # Plan initial guess trajectory button pressed, and at initial guess planning state
        if (
            new_plan_initial_guess_clicks > plan_initial_guess_clicks
            and state == State.INITIAL_GUESS_PLANNING
        ):
            plan_initial_guess_clicks = new_plan_initial_guess_clicks
            print(colored("NEW STATE: PLANNING INITIAL GUESS TRAJECTORY", "cyan"))
            print("Planning trajectory to go to waypoint " + str(path_idx + 1) + "...")

            if path_idx >= len(path_joint_poses) - 1:
                print("Completed all joint poses for hemisphere scanning.")
                continue

            station_context = station.GetMyContextFromRoot(simulator.get_context())
            q_current = station.GetOutputPort("iiwa.position_measured").Eval(
                station_context
            )

            (
                trajopt,
                prog,
                traj_plot_state,
            ) = setup_trajectory_optimization_from_q1_to_q2(
                station=station,
                q1=q_current,
                q2=path_joint_poses[path_idx],
                vel_limits=vel_limits,
                acc_limits=acc_limits,
                duration_constraints=(0.5, 5.0),
                num_control_points=10,
                duration_cost=1.0,
                path_length_cost=1.0,
                visualize_solving=True,
            )

            # Solve for initial guess
            traj_plot_state["rgba"] = Rgba(
                1, 0.5, 0, 1
            )  # Set initial guess color to orange
            result = Solve(prog)
            if not result.is_success():
                print("Trajectory optimization failed, even without collisions!")
                print(result.get_solver_id().name())
            trajopt.SetInitialGuess(trajopt.ReconstructTrajectory(result))

            trajopt = add_collision_constraints_to_trajectory(
                station,
                trajopt,
            )
            state = State.FINAL_PLANNING

        # Plan final trajectory button pressed, and at final planning state
        if new_plan_final_clicks > final_plan_clicks and state == State.FINAL_PLANNING:
            final_plan_clicks = new_plan_final_clicks
            print(colored("NEW STATE: PLANNING FINAL TRAJECTORY", "cyan"))

            traj_plot_state["rgba"] = Rgba(
                0, 1, 0, 1
            )  # Set final trajectory color to green
            result = Solve(prog)
            if not result.is_success():
                print("Trajectory optimization failed")
                print(result.get_solver_id().name())
                continue

            print("Final trajectory optimization succeeded!")

            trajectory = resolve_with_toppra(  # At this point all this is doing is time-optimizing to make the traj as fast as possible
                station,
                trajopt,
                result,
                vel_limits,
                acc_limits,
            )

            print(
                f"✓ TOPPRA succeeded! Trajectory duration: {trajectory.end_time():.2f}s"
            )

            path_idx += 1
            state = State.MOVING

        # Move to goal button pressed and at moving state
        if new_move_clicks > move_clicks and state == State.MOVING:
            move_clicks = new_move_clicks
            if trajectory is None:
                print("No trajectory planned yet!")
            else:
                print("Executing trajectory...")
                executing_trajectory = True
                trajectory_start_time = simulator.get_context().get_time()

        if executing_trajectory:
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
                trajectory = None
                executing_trajectory = False
                state = State.INITIAL_GUESS_PLANNING

        simulator.AdvanceTo(simulator.get_context().get_time() + 0.1)

    station.internal_meshcat.DeleteButton("Stop Simulation")
    station.internal_meshcat.DeleteButton("Plan Initial Guess Trajectory")
    station.internal_meshcat.DeleteButton("Plan Final Trajectory")
    station.internal_meshcat.DeleteButton("Move to Goal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_hardware",
        action="store_true",
        help="Whether to use real world hardware.",
    )

    args = parser.parse_args()
    main(use_hardware=args.use_hardware)
