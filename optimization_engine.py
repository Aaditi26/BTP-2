import json
import random
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from sklearn.ensemble import RandomForestRegressor


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

COST_RATE = 18.0
WEAR_FACTOR = 0.02
DEFAULT_COOLANTS = ["Dry", "Wet", "MQL"]


class OperationSpec:
    def __init__(
        self,
        code: int,
        name: str,
        valid_tools: Sequence[str],
        speed_range: Tuple[float, float],
        feed_range: Tuple[float, float],
        depth_range: Tuple[float, float],
    ) -> None:
        self.code = code
        self.name = name
        self.valid_tools = tuple(valid_tools)
        self.speed_range = speed_range
        self.feed_range = feed_range
        self.depth_range = depth_range


OPERATIONS_LIBRARY: Dict[str, OperationSpec] = {
    "Facing": OperationSpec(1, "Facing", ("Carbide Insert", "HSS"), (90.0, 220.0), (0.08, 0.25), (0.5, 2.5)),
    "Centering": OperationSpec(2, "Centering", ("Center Drill",), (60.0, 180.0), (0.05, 0.18), (0.3, 1.0)),
    "Drilling": OperationSpec(3, "Drilling", ("Twist Drill", "Carbide Drill"), (70.0, 200.0), (0.06, 0.22), (0.8, 3.0)),
    "Turning": OperationSpec(4, "Turning", ("Carbide Insert", "Ceramic Tool"), (100.0, 260.0), (0.1, 0.35), (0.8, 3.5)),
    "Milling": OperationSpec(5, "Milling", ("End Mill", "Face Mill"), (80.0, 240.0), (0.05, 0.2), (0.5, 3.0)),
    "Finishing": OperationSpec(6, "Finishing", ("CBN Tool", "Carbide Insert"), (120.0, 300.0), (0.04, 0.16), (0.2, 1.0)),
}

MATERIAL_SPEED_LIMITS = {
    "Steel": 200.0,
    "Aluminum": 400.0,
    "Cast Iron": 180.0,
    "Titanium": 120.0,
}

MATERIAL_FACTOR = {
    "Steel": 1.25,
    "Aluminum": 0.8,
    "Cast Iron": 1.1,
    "Titanium": 1.55,
}

TOOL_FACTOR = {
    "HSS": 1.18,
    "Center Drill": 1.0,
    "Twist Drill": 1.1,
    "Carbide Drill": 0.92,
    "Carbide Insert": 0.9,
    "Ceramic Tool": 0.82,
    "End Mill": 0.98,
    "Face Mill": 0.95,
    "CBN Tool": 0.78,
}

COOLANT_FACTOR = {
    "Dry": 1.08,
    "Wet": 0.96,
    "MQL": 0.92,
}


def build_training_data(samples: int = 1200) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for _ in range(samples):
        material = random.choice(list(MATERIAL_SPEED_LIMITS))
        machine_type = random.choice(["CNC Lathe", "VMC", "Machining Center"])
        hardness = random.uniform(120.0, 320.0)
        finish_target = random.uniform(0.4, 3.2)

        operation = random.choice(list(OPERATIONS_LIBRARY.values()))
        speed_limit = min(operation.speed_range[1], MATERIAL_SPEED_LIMITS[material])
        speed = random.uniform(operation.speed_range[0], speed_limit)
        feed = random.uniform(*operation.feed_range)
        depth = random.uniform(*operation.depth_range)
        tool = random.choice(list(operation.valid_tools))
        coolant = random.choice(DEFAULT_COOLANTS)

        machine_factor = 0.95 if machine_type == "Machining Center" else 1.0
        hardness_factor = 1.0 + max(0.0, hardness - 180.0) / 500.0
        finish_factor = 1.0 + max(0.0, 1.8 - finish_target) / 8.0
        base_time = (250.0 / speed) + (feed * 7.5) + (depth * 2.5)
        operation_factor = 0.75 + operation.code * 0.14

        time = (
            base_time
            * MATERIAL_FACTOR[material]
            * TOOL_FACTOR[tool]
            * COOLANT_FACTOR[coolant]
            * hardness_factor
            * finish_factor
            * machine_factor
        )
        time += np.random.normal(0, 0.08)

        rows.append(
            {
                "operation_name": operation.name,
                "operation_code": operation.code,
                "material_type": material,
                "material_hardness": hardness,
                "machine_type": machine_type,
                "surface_finish_requirement": finish_target,
                "speed": speed,
                "feed": feed,
                "depth": depth,
                "tool_type": tool,
                "coolant_condition": coolant,
                "machining_time": max(0.2, time * operation_factor),
            }
        )

    return pd.DataFrame(rows)


class MachiningOptimizationEngine:
    def __init__(self, dataset: pd.DataFrame | None = None, seed: int = SEED) -> None:
        self.seed = seed
        self.random = random.Random(seed)
        self.dataset = dataset.copy() if dataset is not None else build_training_data()
        self.feature_columns: List[str] = []
        self.model = self._train_model(self.dataset)
        self._ensure_deap_types()

    def _train_model(self, dataset: pd.DataFrame) -> RandomForestRegressor:
        features = dataset.drop(columns=["machining_time"])
        encoded = pd.get_dummies(features)
        self.feature_columns = encoded.columns.tolist()
        target = dataset["machining_time"]

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_split=4,
            random_state=self.seed,
        )
        model.fit(encoded, target)
        return model

    def _ensure_deap_types(self) -> None:
        if not hasattr(creator, "MachiningFitnessMin"):
            creator.create("MachiningFitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, "MachiningIndividual"):
            creator.create("MachiningIndividual", list, fitness=creator.MachiningFitnessMin)

    def _encode_features(self, row: Dict[str, object]) -> pd.DataFrame:
        encoded = pd.get_dummies(pd.DataFrame([row]))
        return encoded.reindex(columns=self.feature_columns, fill_value=0)

    def predict_operation_metrics(self, job: Dict[str, object], operation_name: str, params: Dict[str, object]) -> Dict[str, float]:
        feature_row = {
            "operation_name": operation_name,
            "operation_code": OPERATIONS_LIBRARY[operation_name].code,
            "material_type": job["material_type"],
            "material_hardness": job["material_hardness"],
            "machine_type": job["machine_type"],
            "surface_finish_requirement": job["surface_finish_requirement"],
            "speed": params["speed"],
            "feed": params["feed"],
            "depth": params["depth"],
            "tool_type": params["tool_type"],
            "coolant_condition": params["coolant_condition"],
        }
        predicted_time = float(self.model.predict(self._encode_features(feature_row))[0])
        predicted_cost = predicted_time * COST_RATE + params["speed"] * 0.03 + params["depth"] * 1.5
        tool_wear = predicted_time * (params["speed"] / 100.0) * (params["depth"] + 0.2) * WEAR_FACTOR
        surface_roughness = max(0.2, (params["feed"] * 7.0) + (params["depth"] * 0.3) - (params["speed"] / 300.0))

        return {
            "machining_time": round(predicted_time, 4),
            "cost": round(predicted_cost, 4),
            "tool_wear": round(tool_wear, 4),
            "surface_roughness": round(surface_roughness, 4),
        }

    def _sequence_length(self, job: Dict[str, object]) -> int:
        return len(job["available_operations"])

    def _create_individual(self, job: Dict[str, object]):
        operation_names = list(job["available_operations"])
        sequence = self.random.sample(operation_names, len(operation_names))
        genes: List[object] = sequence[:]

        for operation_name in operation_names:
            spec = OPERATIONS_LIBRARY[operation_name]
            speed_upper = min(spec.speed_range[1], MATERIAL_SPEED_LIMITS[job["material_type"]], job["machine_limits"]["speed"][1])
            speed_lower = max(spec.speed_range[0], job["machine_limits"]["speed"][0])
            feed_lower = max(spec.feed_range[0], job["machine_limits"]["feed"][0])
            feed_upper = min(spec.feed_range[1], job["machine_limits"]["feed"][1])
            depth_lower = max(spec.depth_range[0], job["machine_limits"]["depth"][0])
            depth_upper = min(spec.depth_range[1], job["machine_limits"]["depth"][1])

            genes.extend(
                [
                    self.random.uniform(speed_lower, speed_upper),
                    self.random.uniform(feed_lower, feed_upper),
                    self.random.uniform(depth_lower, depth_upper),
                    self.random.randrange(len(spec.valid_tools)),
                    self.random.randrange(len(DEFAULT_COOLANTS)),
                    operation_name,
                ]
            )

        return creator.MachiningIndividual(genes)

    def _extract_params(self, individual, job: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        n = self._sequence_length(job)
        params: Dict[str, Dict[str, object]] = {}

        for i, operation_name in enumerate(job["available_operations"]):
            spec = OPERATIONS_LIBRARY[operation_name]
            idx = n + i * 6
            tool_idx = int(round(individual[idx + 3])) % len(spec.valid_tools)
            coolant_idx = int(round(individual[idx + 4])) % len(DEFAULT_COOLANTS)

            params[operation_name] = {
                "speed": float(individual[idx]),
                "feed": float(individual[idx + 1]),
                "depth": float(individual[idx + 2]),
                "tool_type": spec.valid_tools[tool_idx],
                "coolant_condition": DEFAULT_COOLANTS[coolant_idx],
            }

        return params

    def _repair(self, individual, job: Dict[str, object]):
        operation_names = list(job["available_operations"])
        n = len(operation_names)

        repaired_sequence = []
        for name in individual[:n]:
            if name in operation_names and name not in repaired_sequence:
                repaired_sequence.append(name)
        for name in operation_names:
            if name not in repaired_sequence:
                repaired_sequence.append(name)
        individual[:n] = repaired_sequence

        for i, operation_name in enumerate(operation_names):
            spec = OPERATIONS_LIBRARY[operation_name]
            idx = n + i * 6

            speed_upper = min(spec.speed_range[1], MATERIAL_SPEED_LIMITS[job["material_type"]], job["machine_limits"]["speed"][1])
            speed_lower = max(spec.speed_range[0], job["machine_limits"]["speed"][0])
            feed_lower = max(spec.feed_range[0], job["machine_limits"]["feed"][0])
            feed_upper = min(spec.feed_range[1], job["machine_limits"]["feed"][1])
            depth_lower = max(spec.depth_range[0], job["machine_limits"]["depth"][0])
            depth_upper = min(spec.depth_range[1], job["machine_limits"]["depth"][1])

            individual[idx] = float(np.clip(individual[idx], speed_lower, speed_upper))
            individual[idx + 1] = float(np.clip(individual[idx + 1], feed_lower, feed_upper))
            individual[idx + 2] = float(np.clip(individual[idx + 2], depth_lower, depth_upper))
            individual[idx + 3] = int(round(individual[idx + 3])) % len(spec.valid_tools)
            individual[idx + 4] = int(round(individual[idx + 4])) % len(DEFAULT_COOLANTS)
            individual[idx + 5] = operation_name

        return individual

    def _precedence_penalty(self, sequence: Sequence[str], job: Dict[str, object]) -> float:
        penalty = 0.0
        for pair in job.get("precedence_constraints", []):
            if sequence.index(pair["before"]) > sequence.index(pair["after"]):
                penalty += 100.0
        for fixed in job.get("fixed_positions", []):
            required_index = fixed["index"]
            if sequence[required_index] != fixed["operation"]:
                penalty += 100.0
        return penalty

    def _evaluate(self, individual, job: Dict[str, object]):
        individual = self._repair(individual, job)
        sequence = individual[: self._sequence_length(job)]
        params_by_operation = self._extract_params(individual, job)

        total_time = 0.0
        total_cost = 0.0
        total_wear = 0.0

        for operation_name in sequence:
            metrics = self.predict_operation_metrics(job, operation_name, params_by_operation[operation_name])
            total_time += metrics["machining_time"]
            total_cost += metrics["cost"]
            total_wear += metrics["tool_wear"]

        penalty = self._precedence_penalty(sequence, job)
        return total_time + penalty, total_cost + penalty, total_wear + penalty

    def _mate(self, ind1, ind2, job: Dict[str, object]):
        n = self._sequence_length(job)
        seq1 = ind1[:n]
        seq2 = ind2[:n]

        left, right = sorted(self.random.sample(range(n), 2))
        slice1 = seq1[left:right]
        slice2 = seq2[left:right]

        child1 = [name for name in seq2 if name not in slice1]
        child2 = [name for name in seq1 if name not in slice2]

        child1[left:left] = slice1
        child2[left:left] = slice2

        ind1[:n] = child1[:n]
        ind2[:n] = child2[:n]

        for i in range(n, len(ind1)):
            if (i - n) % 6 == 5:
                continue
            if self.random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

        return self._repair(ind1, job), self._repair(ind2, job)

    def _mutate(self, individual, job: Dict[str, object]):
        n = self._sequence_length(job)

        if self.random.random() < 0.45:
            left, right = self.random.sample(range(n), 2)
            individual[left], individual[right] = individual[right], individual[left]

        for i, operation_name in enumerate(job["available_operations"]):
            spec = OPERATIONS_LIBRARY[operation_name]
            idx = n + i * 6

            if self.random.random() < 0.30:
                individual[idx] += self.random.uniform(-18.0, 18.0)
            if self.random.random() < 0.30:
                individual[idx + 1] += self.random.uniform(-0.03, 0.03)
            if self.random.random() < 0.30:
                individual[idx + 2] += self.random.uniform(-0.4, 0.4)
            if self.random.random() < 0.20:
                individual[idx + 3] = self.random.randrange(len(spec.valid_tools))
            if self.random.random() < 0.20:
                individual[idx + 4] = self.random.randrange(len(DEFAULT_COOLANTS))

        return (self._repair(individual, job),)

    def _build_toolbox(self, job: Dict[str, object]) -> base.Toolbox:
        toolbox = base.Toolbox()
        toolbox.register("individual", self._create_individual, job)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate, job=job)
        toolbox.register("mate", self._mate, job=job)
        toolbox.register("mutate", self._mutate, job=job)
        toolbox.register("select", tools.selNSGA2)
        return toolbox

    def _decode_solution(self, individual, job: Dict[str, object]) -> Dict[str, object]:
        n = self._sequence_length(job)
        sequence = individual[:n]
        params_by_operation = self._extract_params(individual, job)

        operation_results = []
        total_time = 0.0
        total_cost = 0.0
        total_wear = 0.0

        for operation_name in sequence:
            params = params_by_operation[operation_name]
            metrics = self.predict_operation_metrics(job, operation_name, params)
            total_time += metrics["machining_time"]
            total_cost += metrics["cost"]
            total_wear += metrics["tool_wear"]

            operation_results.append(
                {
                    "operation": operation_name,
                    "parameters": {
                        "cutting_speed": round(params["speed"], 4),
                        "feed_rate": round(params["feed"], 4),
                        "depth_of_cut": round(params["depth"], 4),
                        "tool_type": params["tool_type"],
                        "coolant_condition": params["coolant_condition"],
                    },
                    "predictions": metrics,
                }
            )

        return {
            "sequence": sequence,
            "operations": operation_results,
            "total_machining_time": round(total_time, 4),
            "total_cost": round(total_cost, 4),
            "total_tool_wear": round(total_wear, 4),
        }

    def optimize(self, job: Dict[str, object], population_size: int = 36, generations: int = 18) -> Dict[str, object]:
        toolbox = self._build_toolbox(job)
        population = toolbox.population(n=population_size)

        for individual in population:
            individual.fitness.values = toolbox.evaluate(individual)

        for _ in range(generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=0.3)
            for individual in offspring:
                individual.fitness.values = toolbox.evaluate(individual)
            population = toolbox.select(population + offspring, k=len(population))

        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        decoded = [self._decode_solution(ind, job) for ind in pareto_front]
        decoded.sort(key=lambda item: (item["total_machining_time"], item["total_cost"], item["total_tool_wear"]))

        unique_decoded = []
        seen = set()
        for item in decoded:
            signature = (
                tuple(item["sequence"]),
                tuple(
                    (
                        op["operation"],
                        op["parameters"]["tool_type"],
                        op["parameters"]["coolant_condition"],
                        round(op["parameters"]["cutting_speed"], 3),
                        round(op["parameters"]["feed_rate"], 3),
                        round(op["parameters"]["depth_of_cut"], 3),
                    )
                    for op in item["operations"]
                ),
            )
            if signature not in seen:
                seen.add(signature)
                unique_decoded.append(item)

        best = unique_decoded[0]
        return {
            "job_id": job["job_id"],
            "material_type": job["material_type"],
            "machine_type": job["machine_type"],
            "objective": "Minimize machining time, cost, and tool wear",
            "optimal_sequence": best["sequence"],
            "optimal_parameters": best["operations"],
            "minimum_total_machining_time": best["total_machining_time"],
            "minimum_total_cost": best["total_cost"],
            "minimum_total_tool_wear": best["total_tool_wear"],
            "pareto_optimal_set": unique_decoded[:5],
        }


def sample_job_input() -> Dict[str, object]:
    return {
        "job_id": "JOB-101",
        "material_type": "Steel",
        "material_hardness": 240.0,
        "workpiece_geometry": {"length": 120.0, "diameter": 45.0, "thickness": 18.0},
        "tolerance_requirement": 0.02,
        "surface_finish_requirement": 1.2,
        "machine_type": "CNC Lathe",
        "available_operations": ["Facing", "Centering", "Drilling", "Turning", "Finishing"],
        "precedence_constraints": [
            {"before": "Facing", "after": "Centering"},
            {"before": "Centering", "after": "Drilling"},
            {"before": "Turning", "after": "Finishing"},
        ],
        "fixed_positions": [
            {"operation": "Facing", "index": 0},
            {"operation": "Finishing", "index": 4}
        ],
        "machine_limits": {
            "speed": (80.0, 260.0),
            "feed": (0.04, 0.35),
            "depth": (0.2, 3.5),
        },
    }


def optimize_job(job_input: Dict[str, object]) -> Dict[str, object]:
    engine = MachiningOptimizationEngine()
    return engine.optimize(job_input)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as handle:
            job_data = json.load(handle)
    else:
        job_data = sample_job_input()

    result = optimize_job(job_data)
    print(json.dumps(result, indent=2))
