"""Genetic algorithm optimization engine for router placement."""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from uuid import UUID
import random
import copy

from app.schemas.project import MapData
from app.schemas.router import Router as RouterSchema
from app.schemas.optimization import OptimizationConstraints, RouterPlacement
from app.services.rf_propagation import get_rf_engine, SignalGrid, calculate_coverage_percentage


@dataclass
class Individual:
    """Represents a solution (chromosome) in the genetic algorithm."""
    router_positions: List[Tuple[float, float]] = field(default_factory=list)
    router_ids: List[UUID] = field(default_factory=list)
    fitness: float = 0.0
    coverage: float = 0.0
    cost: float = 0.0

    def copy(self) -> 'Individual':
        """Create a deep copy of this individual."""
        return Individual(
            router_positions=copy.deepcopy(self.router_positions),
            router_ids=copy.deepcopy(self.router_ids),
            fitness=self.fitness,
            coverage=self.coverage,
            cost=self.cost
        )


class GeneticOptimizer:
    """
    Genetic algorithm for optimizing router placement.

    Chromosome: List of (x, y, router_id) tuples
    Fitness: Weighted combination of coverage, cost, uniformity
    """

    def __init__(
        self,
        map_data: MapData,
        available_routers: List[RouterSchema],
        constraints: OptimizationConstraints,
        scale: float,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5
    ):
        self.map_data = map_data
        self.available_routers = available_routers
        self.constraints = constraints
        self.scale = scale

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

        self.rf_engine = get_rf_engine()

        # Map dimensions
        self.width = map_data.dimensions.width
        self.height = map_data.dimensions.height

        # Compute valid placement area
        self.valid_mask = self._compute_valid_area()

        # Router lookup by ID
        self.router_map = {r.id: r for r in available_routers}

    def optimize(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> List[Individual]:
        """
        Run genetic algorithm optimization.

        Args:
            progress_callback: Optional callback for progress updates
                              (generation, total_generations, best_fitness)

        Returns:
            List of best solutions (Pareto front)
        """
        # Initialize population
        population = self._initialize_population()

        # Evaluate initial fitness
        for individual in population:
            self._evaluate_fitness(individual)

        best_solutions = []
        best_fitness_history = []

        # Evolution loop
        for generation in range(self.generations):
            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Track best
            best = population[0]
            best_fitness_history.append(best.fitness)

            # Progress callback
            if progress_callback:
                progress_callback(generation, self.generations, best.fitness)

            # Selection
            parents = self._selection(population)

            # Crossover
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parents[i], parents[i + 1])
                    offspring.extend([child1, child2])
                else:
                    offspring.extend([parents[i].copy(), parents[i + 1].copy()])

            # Mutation
            for individual in offspring:
                if random.random() < self.mutation_rate:
                    self._mutate(individual)

            # Evaluate offspring
            for individual in offspring:
                self._evaluate_fitness(individual)

            # Elitism: keep best from previous generation
            elite = population[:self.elite_size]

            # New population
            offspring.sort(key=lambda x: x.fitness, reverse=True)
            population = elite + offspring[:self.population_size - self.elite_size]

        # Sort final population
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Return top diverse solutions
        return self._get_diverse_solutions(population, count=5)

    def _initialize_population(self) -> List[Individual]:
        """Create initial random population."""
        population = []

        for _ in range(self.population_size):
            # Determine number of routers
            max_routers = self.constraints.max_routers or 10
            min_routers = 1
            num_routers = random.randint(min_routers, max_routers)

            # Random positions in valid area
            positions = []
            for _ in range(num_routers):
                pos = self._random_valid_position()
                positions.append(pos)

            # Random router selection (respecting allowed_router_ids)
            available = self.available_routers
            if self.constraints.allowed_router_ids:
                available = [
                    r for r in available
                    if r.id in self.constraints.allowed_router_ids
                ]

            if not available:
                available = self.available_routers

            router_ids = [
                random.choice(available).id
                for _ in range(num_routers)
            ]

            individual = Individual(
                router_positions=positions,
                router_ids=router_ids
            )
            population.append(individual)

        return population

    def _evaluate_fitness(self, individual: Individual) -> float:
        """
        Calculate fitness score for an individual.

        Fitness = weighted combination of:
        - Coverage percentage (maximize)
        - Cost (minimize)
        - Signal uniformity (maximize)
        - Number of routers (minimize)
        """
        if not individual.router_positions:
            individual.fitness = -1000.0
            return individual.fitness

        # Get router specs
        routers = [
            self.router_map.get(rid)
            for rid in individual.router_ids
        ]
        routers = [r for r in routers if r is not None]

        if not routers:
            individual.fitness = -1000.0
            return individual.fitness

        # Predict signal coverage
        # Use coarser grid for large images to speed up optimization
        img_size = self.map_data.dimensions.width * self.map_data.dimensions.height
        if img_size > 1000000:  # > 1 megapixel
            grid_res = 20  # Very coarse for large images
        elif img_size > 500000:
            grid_res = 10
        else:
            grid_res = 4

        signal_grid = self.rf_engine.predict_signal(
            self.map_data,
            individual.router_positions,
            routers,
            self.scale,
            grid_resolution=grid_res
        )

        # Coverage percentage
        coverage = calculate_coverage_percentage(
            signal_grid,
            self.constraints.min_signal_strength_dbm
        )
        individual.coverage = coverage

        # Cost
        total_cost = sum(float(r.price_usd or 0) for r in routers)
        individual.cost = total_cost

        max_budget = self.constraints.max_budget or 10000
        cost_ratio = 1.0 - min(total_cost / max_budget, 1.0)

        # Signal uniformity (lower std dev is better)
        covered_signals = signal_grid.grid[
            signal_grid.grid > self.constraints.min_signal_strength_dbm
        ]
        if len(covered_signals) > 0:
            uniformity = 1.0 / (np.std(covered_signals) + 1.0)
        else:
            uniformity = 0.0

        # Router count penalty (prefer fewer routers)
        max_routers = self.constraints.max_routers or 10
        router_penalty = 1.0 - (len(routers) / max_routers)

        # Weighted fitness
        if self.constraints.prioritize_cost:
            # Cost-focused weights
            fitness = (
                0.4 * (coverage / 100.0) +
                0.35 * cost_ratio +
                0.15 * uniformity +
                0.1 * router_penalty
            )
        else:
            # Coverage-focused weights (default)
            fitness = (
                0.5 * (coverage / 100.0) +
                0.2 * cost_ratio +
                0.2 * uniformity +
                0.1 * router_penalty
            )

        # Hard constraint penalties
        if coverage < self.constraints.min_coverage_percent:
            # Penalty proportional to coverage gap
            gap = self.constraints.min_coverage_percent - coverage
            fitness -= gap * 0.1

        if total_cost > max_budget:
            # Penalty for exceeding budget
            excess = (total_cost - max_budget) / max_budget
            fitness -= excess * 5.0

        individual.fitness = fitness
        return fitness

    def _selection(self, population: List[Individual]) -> List[Individual]:
        """
        Tournament selection.

        Select parents for reproduction using tournament selection.
        """
        selected = []
        tournament_size = 3

        for _ in range(len(population)):
            # Random tournament
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner.copy())

        return selected

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Uniform crossover between two parents.

        Randomly swaps router placements between parents.
        """
        # Determine child sizes
        size1 = len(parent1.router_positions)
        size2 = len(parent2.router_positions)
        avg_size = (size1 + size2) // 2

        # Mix router positions and types
        all_positions = parent1.router_positions + parent2.router_positions
        all_ids = parent1.router_ids + parent2.router_ids

        # Randomly select for each child
        indices = list(range(len(all_positions)))
        random.shuffle(indices)

        child1_size = random.randint(1, min(avg_size + 2, len(indices)))
        child2_size = random.randint(1, min(avg_size + 2, len(indices)))

        child1_indices = indices[:child1_size]
        child2_indices = indices[child1_size:child1_size + child2_size]

        child1 = Individual(
            router_positions=[all_positions[i] for i in child1_indices],
            router_ids=[all_ids[i] for i in child1_indices]
        )

        child2 = Individual(
            router_positions=[all_positions[i] for i in child2_indices],
            router_ids=[all_ids[i] for i in child2_indices]
        )

        return child1, child2

    def _mutate(self, individual: Individual):
        """
        Mutate an individual.

        Possible mutations:
        - Move a router to new position
        - Change router type
        - Add/remove a router
        """
        if not individual.router_positions:
            # Add a router if empty
            x, y = self._random_valid_position()
            individual.router_positions.append((x, y))
            individual.router_ids.append(random.choice(self.available_routers).id)
            return

        mutation_type = random.choice(['move', 'change_type', 'add_remove'])

        if mutation_type == 'move':
            # Move random router to nearby position
            idx = random.randint(0, len(individual.router_positions) - 1)
            old_x, old_y = individual.router_positions[idx]

            # Move within a radius
            radius = min(self.width, self.height) * 0.2
            new_x = old_x + random.uniform(-radius, radius)
            new_y = old_y + random.uniform(-radius, radius)

            # Clamp to bounds
            new_x = max(10, min(self.width - 10, new_x))
            new_y = max(10, min(self.height - 10, new_y))

            individual.router_positions[idx] = (new_x, new_y)

        elif mutation_type == 'change_type':
            # Change random router type
            idx = random.randint(0, len(individual.router_ids) - 1)
            available = self.available_routers
            if self.constraints.allowed_router_ids:
                available = [
                    r for r in available
                    if r.id in self.constraints.allowed_router_ids
                ]
            if available:
                individual.router_ids[idx] = random.choice(available).id

        elif mutation_type == 'add_remove':
            max_routers = self.constraints.max_routers or 10

            # 50% chance to add, 50% to remove
            if random.random() < 0.5 and len(individual.router_positions) < max_routers:
                # Add router
                x, y = self._random_valid_position()
                individual.router_positions.append((x, y))
                individual.router_ids.append(random.choice(self.available_routers).id)
            elif len(individual.router_positions) > 1:
                # Remove router
                idx = random.randint(0, len(individual.router_positions) - 1)
                del individual.router_positions[idx]
                del individual.router_ids[idx]

    def _get_diverse_solutions(
        self,
        population: List[Individual],
        count: int = 5
    ) -> List[Individual]:
        """
        Extract diverse top solutions.

        Ensures solutions are meaningfully different from each other.
        """
        if len(population) <= count:
            return population

        selected = [population[0]]  # Best solution

        for candidate in population[1:]:
            if len(selected) >= count:
                break

            # Check if candidate is different enough
            is_diverse = True
            for existing in selected:
                similarity = self._solution_similarity(candidate, existing)
                if similarity > 0.8:  # Too similar
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(candidate)

        # Fill remaining slots with next best
        while len(selected) < count and len(selected) < len(population):
            for candidate in population:
                if candidate not in selected:
                    selected.append(candidate)
                    break

        return selected

    def _solution_similarity(
        self,
        ind1: Individual,
        ind2: Individual
    ) -> float:
        """Calculate similarity between two solutions (0-1)."""
        # Compare number of routers
        size_diff = abs(len(ind1.router_positions) - len(ind2.router_positions))
        size_sim = 1.0 - size_diff / max(len(ind1.router_positions), len(ind2.router_positions), 1)

        # Compare positions (average distance)
        if not ind1.router_positions or not ind2.router_positions:
            return size_sim

        min_distances = []
        for pos1 in ind1.router_positions:
            min_dist = float('inf')
            for pos2 in ind2.router_positions:
                dist = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                min_dist = min(min_dist, dist)
            min_distances.append(min_dist)

        avg_dist = np.mean(min_distances)
        max_dist = np.sqrt(self.width ** 2 + self.height ** 2)
        pos_sim = 1.0 - min(avg_dist / max_dist, 1.0)

        return 0.5 * size_sim + 0.5 * pos_sim

    def _compute_valid_area(self) -> np.ndarray:
        """
        Compute mask of valid router placement positions.

        Routers cannot be placed:
        - Inside walls
        - In forbidden zones (bathrooms, etc.)
        - Outside the floor boundary
        """
        mask = np.zeros((self.height, self.width), dtype=bool)

        # If rooms are detected, only allow placement inside rooms
        if self.map_data.rooms:
            mask_uint8 = mask.astype(np.uint8)
            for room in self.map_data.rooms:
                if room.polygon and len(room.polygon) >= 3:
                    # Fill room polygon as valid
                    pts = np.array(room.polygon, dtype=np.int32)
                    cv2.fillPoly(mask_uint8, [pts], 1)
            mask = mask_uint8.astype(bool)
        else:
            # Fallback: use bounding box of all walls with padding
            if self.map_data.walls:
                all_x = []
                all_y = []
                for wall in self.map_data.walls:
                    all_x.extend([wall.start.x, wall.end.x])
                    all_y.extend([wall.start.y, wall.end.y])

                min_x = max(0, int(min(all_x)) + 50)
                max_x = min(self.width, int(max(all_x)) - 50)
                min_y = max(0, int(min(all_y)) + 50)
                max_y = min(self.height, int(max(all_y)) - 50)

                mask[min_y:max_y, min_x:max_x] = True
            else:
                # No walls detected, use center region
                margin = int(min(self.width, self.height) * 0.2)
                mask[margin:self.height-margin, margin:self.width-margin] = True

        # Mark walls as invalid (with buffer)
        for wall in self.map_data.walls:
            x1, y1 = int(wall.start.x), int(wall.start.y)
            x2, y2 = int(wall.end.x), int(wall.end.y)

            # Create line with buffer
            # Simple approach: mark rectangle around wall
            buffer = 20  # pixels
            min_x = max(0, min(x1, x2) - buffer)
            max_x = min(self.width, max(x1, x2) + buffer)
            min_y = max(0, min(y1, y2) - buffer)
            max_y = min(self.height, max(y1, y2) + buffer)

            mask[min_y:max_y, min_x:max_x] = False

        # Mark forbidden zones as invalid
        for zone in self.map_data.forbidden_zones:
            # Simple bounding box approach
            if zone.polygon:
                xs = [p[0] for p in zone.polygon]
                ys = [p[1] for p in zone.polygon]
                min_x = max(0, int(min(xs)))
                max_x = min(self.width, int(max(xs)))
                min_y = max(0, int(min(ys)))
                max_y = min(self.height, int(max(ys)))
                mask[min_y:max_y, min_x:max_x] = False

        return mask

    def _random_valid_position(self) -> Tuple[float, float]:
        """Get random valid (x, y) position for router placement."""
        # Simple approach: random position with margin from edges
        margin = 30  # pixels from edge

        for _ in range(100):  # Max attempts
            x = random.uniform(margin, self.width - margin)
            y = random.uniform(margin, self.height - margin)

            # Check if valid
            ix, iy = int(x), int(y)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                if self.valid_mask[iy, ix]:
                    return (x, y)

        # Fallback: center of floor plan
        return (self.width / 2, self.height / 2)
