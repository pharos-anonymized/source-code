# type: ignore
# Pyomo has some type issues, so we extract the solver logic to a single file

import logging

import numpy as np
import pyomo.environ as pyo
from pyomo.environ import value
from termcolor import colored

from core.env import Env, dis_to_cube


def print_model_info(model: pyo.ConcreteModel):
    logging.debug("===== Variables =====")
    for v in model.component_data_objects(pyo.Var, descend_into=True):
        val = v.value
        if val is None:
            logging.debug(f"Variable {v.name} Index: {v.index()} has no initial value.")
        else:
            logging.debug(f"Variable {v.name} Index: {v.index()} Value: {val}")

    logging.debug("\n===== Constraints =====")
    for c in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        logging.debug(f"Constraint {c.name} Index: {c.index()} Expression: {c.expr}")
        logging.debug(f"  Lower bound: {c.lower}, Upper bound: {c.upper}")

    # Print total number of constraints and variables
    num_constraints = sum(1 for _ in model.component_data_objects(pyo.Constraint))
    num_vars = sum(1 for _ in model.component_data_objects(pyo.Var))
    logging.debug(colored(f"\nTotal Constraints: {num_constraints}, Total Variables: {num_vars}", "yellow"))


class ObjectiveFunctions:
    """Objective functions for the optimization problem"""

    def __init__(self, env: Env):
        self.env = env

        self.current_pos = [agent.position for agent in env.agents]
        self.target_pos = [agent.target_pos for agent in env.agents]
        self.velocities = [agent.velocity for agent in env.agents]

        self.human_positions = np.array([h.position for h in env.humans])
        self.human_positions[:, 1] = 1.7
        self.human_velocities = np.array([h.velocity for h in env.humans])
        self.scare_factors = np.array([env.scare_factor for h in env.humans])

    def closer_reward(self, model: pyo.ConcreteModel):
        """Calculate closer reward"""
        return (
            pyo.quicksum(
                # Distance from original position to target minus distance from new position to target, positive value means closer to target
                -pyo.sqrt(pyo.quicksum((model.next_pos[n, d] - self.target_pos[n][d]) ** 2 for d in range(3)))
                + pyo.sqrt(pyo.quicksum((self.current_pos[n][d] - self.target_pos[n][d]) ** 2 for d in range(3)))
                for n in model.N
            )
            * self.env.closer_factor
        )

    def reach_reward(self, model: pyo.ConcreteModel):
        """Calculate reach reward"""
        k = 1000
        return self.env.reach_factor * pyo.quicksum(
            # Use exponential function to approximate boolean judgment:
            # when closer to target, exponential term is smaller and function value approaches 1;
            # conversely when farther, function value approaches 0
            pyo.exp(-k * pyo.quicksum((model.next_pos[n, d] - self.target_pos[n][d]) ** 2 for d in range(3)))
            for n in model.N
        )

    def scare_penalty(self, model: pyo.ConcreteModel):
        """Calculate scare penalty"""
        fearness_sum = 0.0

        for i in model.N:
            for j in range(len(self.env.humans)):
                # Skip if next moment cannot enter fear distance
                if np.linalg.norm(self.current_pos[i] - self.human_positions[j]) > self.env.cutoff_scare_distance + 2:
                    continue

                # Calculate distance between agent and human
                distance_sq = pyo.quicksum((model.next_pos[i, k] - self.human_positions[j][k]) ** 2 for k in range(3))
                distance = pyo.sqrt(distance_sq)

                # Use action multiplied by max speed as agent velocity
                agent_velocity = [model.X[i, k] * self.env.agent_max_speed for k in range(3)]

                # Calculate velocity norm
                agent_velocity_norm = pyo.sqrt(pyo.quicksum(agent_velocity[k] ** 2 for k in range(3)) + 1e-6)
                human_velocity_norm = np.linalg.norm(self.human_velocities[j])

                # Calculate dot product of velocities and cosine similarity, add epsilon to avoid division by zero
                dot_product = pyo.quicksum(self.human_velocities[j][k] * agent_velocity[k] for k in range(3))
                cosine_v = dot_product / (human_velocity_norm * agent_velocity_norm + 1e-6)

                # Calculate relative position and velocity
                rel_pos = [model.next_pos[i, k] - self.human_positions[j][k] for k in range(3)]
                rel_vel = [agent_velocity[k] - self.human_velocities[j][k] for k in range(3)]
                rel_pos_norm = pyo.sqrt(pyo.quicksum(rel_pos[k] ** 2 for k in range(3)))
                rel_vel_norm = pyo.sqrt(pyo.quicksum(rel_vel[k] ** 2 for k in range(3)))

                # Calculate cosine similarity of relative velocity and position, add epsilon to avoid division by zero
                dot_product = pyo.quicksum(rel_vel[k] * rel_pos[k] for k in range(3))
                cosine_r = dot_product / (rel_vel_norm * rel_pos_norm + 1e-6)

                # Use smooth approximation for max(0, cosine_r)
                # max(0, x) ≈ 0.5 * (x + sqrt(x^2 + ε))
                pos_angle_factor = 0.5 * (cosine_r + pyo.sqrt(cosine_r**2 + 1e-6))

                # Use sigmoid function to approximate truncation
                # When distance is greater than cutoff_scare_distance, cutoff_factor approaches 0
                # When distance is less than cutoff_scare_distance, cutoff_factor approaches 1
                steepness = 100
                sigmoid_input = steepness * (self.env.cutoff_scare_distance - distance)
                cutoff_factor = 1 / (1 + pyo.exp(-sigmoid_input))

                # Comprehensively calculate fear value and apply truncation
                fearness = (1 / (distance + 1e-6)) * (1 - cosine_v) * pos_angle_factor
                fearness_sum += cutoff_factor * fearness

        return fearness_sum * self.env.scare_factor

    def collision_penalty(self, model: pyo.ConcreteModel):
        """Calculate collision penalty of buildings and agent-to-human collisions"""

        def soft_box_indicator(position, bbox_min, bbox_max, steepness):
            # Left boundary sigmoid function: approaches 1 when position > bbox_min
            left_sigmoid = 1 / (1 + pyo.exp(-steepness * (position - bbox_min)))
            # Right boundary sigmoid function: approaches 1 when position < bbox_max
            right_sigmoid = 1 / (1 + pyo.exp(steepness * (position - bbox_max)))
            # Product of two sigmoids approaches 1 within [bbox_min, bbox_max] interval, approaches 0 outside
            return left_sigmoid * right_sigmoid

        collision_count = 0
        safe_distance = 0.5  # Safe distance

        # Agent-Building collisions
        for n in model.N:
            for i, building in enumerate(self.env.buildings):
                # Skip buildings that are too far away to improve efficiency
                if dis_to_cube(self.current_pos[n], building.bbox) > 2:
                    continue

                # Expand building bounding box by adding safe distance
                bbox_min = building.bbox[:3] - safe_distance
                bbox_max = building.bbox[3:] + safe_distance

                # Steepness parameter for sigmoid function, higher value closer to hard truncation
                steepness = 10

                # Calculate the degree to which the agent is inside the building in each dimension
                # Use double sigmoid function to create a smooth function that is 1 within [min, max] interval and 0 outside
                inside_x = soft_box_indicator(model.next_pos[n, 0], bbox_min[0], bbox_max[0], steepness)
                inside_y = soft_box_indicator(model.next_pos[n, 1], bbox_min[1], bbox_max[1], steepness)
                inside_z = soft_box_indicator(model.next_pos[n, 2], bbox_min[2], bbox_max[2], steepness)

                # Use product to approximate AND operation: only counts as collision when inside building in all dimensions
                collision_indicator = inside_x * inside_y * inside_z
                collision_count += collision_indicator

        # Agent-Human collisions
        for n in model.N:
            for j in range(len(self.env.humans)):
                # Skip humans that are too far away to improve efficiency
                if np.linalg.norm(self.current_pos[n] - self.human_positions[j]) > 2:
                    continue

                # Calculate distance between agent and human
                distance_sq = pyo.quicksum((model.next_pos[n, k] - self.human_positions[j][k]) ** 2 for k in range(3))
                distance = pyo.sqrt(distance_sq)

                # Use sigmoid function to create smooth collision indicator
                # When distance < safe_distance, collision_indicator approaches 1
                # When distance > safe_distance, collision_indicator approaches 0
                steepness = 10
                sigmoid_input = steepness * (safe_distance - distance)
                collision_indicator = 1 / (1 + pyo.exp(-sigmoid_input))
                collision_count += collision_indicator

        # Agent-Agent collisions
        for i in model.N:
            for j in model.N:
                # Avoid double counting and self-collision
                if i >= j:
                    continue

                # Skip agents that are too far away to improve efficiency
                if np.linalg.norm(self.current_pos[i] - self.current_pos[j]) > 2:
                    continue

                # Calculate distance between agents
                distance_sq = pyo.quicksum((model.next_pos[i, k] - model.next_pos[j, k]) ** 2 for k in range(3))
                distance = pyo.sqrt(distance_sq)

                # Use sigmoid function to create smooth collision indicator
                steepness = 10
                sigmoid_input = steepness * (safe_distance - distance)
                collision_indicator = 1 / (1 + pyo.exp(-sigmoid_input))
                collision_count += collision_indicator

        return collision_count * self.env.collision_factor


class ModelBuilder:
    """Builds the Pyomo model with variables and constraints"""

    def __init__(self, env: Env):
        self.env = env

    def create_model(self):
        """Create and configure the Pyomo model"""
        model = pyo.ConcreteModel()

        # Create parameters and sets
        model.dimension = pyo.Param(initialize=3)
        model.N = pyo.RangeSet(0, len(self.env.agents) - 1)

        self._add_action_variables(model)
        self._add_position_expressions(model)

        return model

    def _add_action_variables(self, model):
        """Add decision variables for agent actions"""

        if self.env.action_discrete:
            directions = {
                0: [0, 0, 0],  # No action (stay still)
                1: [1, 0, 0],  # +X
                2: [-1, 0, 0],  # -X
                3: [0, 1, 0],  # +Y
                4: [0, -1, 0],  # -Y
                5: [0, 0, 1],  # +Z
                6: [0, 0, -1],  # -Z
            }

            model.Y = pyo.Var(model.N, range(7), domain=pyo.Binary)
            for n in model.N:
                model.Y[n, 0].set_value(1)  # Set "no action" as initial value
                for k in range(1, 7):
                    model.Y[n, k].set_value(0)

            def one_hot_rule(model, n):
                return pyo.quicksum(model.Y[n, k] for k in range(7)) == 1

            model.index_one_hot_constr = pyo.Constraint(model.N, rule=one_hot_rule)

            def X_rule(model, n, d):
                return pyo.quicksum(directions[k][d] * model.Y[n, k] for k in range(7))

            model.X = pyo.Expression(model.N, range(3), rule=X_rule)
        else:
            model.X = pyo.Var(model.N, range(3), domain=pyo.Reals, bounds=(-1, 1))
            for n in model.N:
                for d in range(3):
                    model.X[n, d].set_value(0.0)

            def unit_vector_constraint(model, n):
                return pyo.quicksum(model.X[n, d] ** 2 for d in range(3)) <= 1

            model.unit_vector_constr = pyo.Constraint(model.N, rule=unit_vector_constraint)

    def _add_position_expressions(self, model):
        """Add expressions for next positions"""

        current_pos = [agent.position for agent in self.env.agents]

        def next_pos_rule(model, n, d):
            return current_pos[n][d] + model.X[n, d]

        model.next_pos = pyo.Expression(model.N, range(3), rule=next_pos_rule)


class PyomoSolver:
    """A solver that uses mathematical methods to solve a single step."""

    def __init__(self, solver_name: str):
        self.solver = pyo.SolverFactory(solver_name)

        self._add_solver_options(solver_name)

    def _add_solver_options(self, solver_name: str):
        if solver_name == "ipopt":
            self.solver.options["tol"] = 1e-6
            self.solver.options["max_iter"] = 500
            self.solver.options["linear_solver"] = "mumps"
            self.solver.options["halt_on_ampl_error"] = "yes"

    def solve_single_step(self, env: Env):
        """Solve for optimal actions for a single time step"""
        # Build model
        model_builder = ModelBuilder(env)
        model = model_builder.create_model()

        # Create objective functions
        obj_funcs = ObjectiveFunctions(env)

        closer_reward = obj_funcs.closer_reward(model)
        scare_penalty = obj_funcs.scare_penalty(model)
        reach_reward = obj_funcs.reach_reward(model)
        building_penalty = obj_funcs.collision_penalty(model)

        # Set objective
        model.objective = pyo.Objective(
            expr=closer_reward - scare_penalty + reach_reward - building_penalty,
            sense=pyo.maximize,
        )

        print_model_info(model)

        # Solve
        results = self.solver.solve(model, tee=logging.getLogger().getEffectiveLevel() <= logging.DEBUG)

        objective = pyo.value(model.objective)
        steps = np.array([value(model.X[n, d]) for n in model.N for d in range(3)]).reshape((len(model.N), 3))

        success = (
            results.solver.termination_condition == pyo.TerminationCondition.optimal
            or results.solver.termination_condition == pyo.TerminationCondition.feasible
            or results.solver.termination_condition == pyo.TerminationCondition.maxIterations  # timeout but feasible
        )

        logging.info(f"Closer reward: {round(pyo.value(closer_reward), 4)}")
        logging.info(f"Scare penalty: {round(pyo.value(scare_penalty), 4)}")

        reach_reward_value = round(pyo.value(reach_reward), 4)
        color = "green" if reach_reward_value > 10 else None
        logging.info(f"Reach reward: {colored(reach_reward_value, color)}")

        building_penalty_value = round(pyo.value(building_penalty), 4)
        color = "red" if building_penalty_value > 100 else None
        log_func = logging.error if building_penalty_value > 100 else logging.info
        log_func(f"Building penalty: {colored(building_penalty_value, color)}")

        return objective, steps, success, results
