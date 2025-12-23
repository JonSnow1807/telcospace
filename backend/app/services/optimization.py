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

        # Adaptive mutation parameters
        self.initial_mutation_rate = mutation_rate
        self.min_mutation_rate = 0.02
        self.mutation_decay = 0.95

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

            # Calculate adaptive mutation rate (decreases over generations)
            adaptive_mutation = max(
                self.min_mutation_rate,
                self.initial_mutation_rate * (self.mutation_decay ** generation)
            )

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

            # Mutation with adaptive rate
            for individual in offspring:
                if random.random() < adaptive_mutation:
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

        # Apply local search refinement to top solutions
        top_solutions = self._get_diverse_solutions(population, count=5)
        refined_solutions = []
        for solution in top_solutions:
            refined = self._local_search(solution)
            refined_solutions.append(refined)

        # Re-sort after local search
        refined_solutions.sort(key=lambda x: x.fitness, reverse=True)
        return refined_solutions

    def _initialize_population(self) -> List[Individual]:
        """
        Create initial population with intelligent seeding strategies.

        Uses a mix of:
        - 25% corner-based (strategic corner coverage)
        - 25% grid-based (even distribution)
        - 20% centroid-based (room centers if available)
        - 30% random (for diversity)
        """
        population = []

        # Calculate counts for each strategy
        corner_count = self.population_size // 4
        grid_count = self.population_size // 4
        centroid_count = self.population_size // 5
        random_count = self.population_size - corner_count - grid_count - centroid_count

        # Get available routers
        available = self.available_routers
        if self.constraints.allowed_router_ids:
            available = [
                r for r in available
                if r.id in self.constraints.allowed_router_ids
            ]
        if not available:
            available = self.available_routers

        max_routers = self.constraints.max_routers or 10

        # Strategy 1: Corner-based placements (good for room coverage)
        population.extend(self._create_corner_individuals(corner_count, available, max_routers))

        # Strategy 2: Grid-based placements (even distribution)
        population.extend(self._create_grid_individuals(grid_count, available, max_routers))

        # Strategy 3: Centroid-based placements (room centers)
        population.extend(self._create_centroid_individuals(centroid_count, available, max_routers))

        # Strategy 4: Random placements (diversity)
        population.extend(self._create_random_individuals(random_count, available, max_routers))

        return population

    def _create_corner_individuals(
        self,
        count: int,
        available_routers: List[RouterSchema],
        max_routers: int
    ) -> List[Individual]:
        """Create individuals with routers near room corners for maximum coverage."""
        individuals = []

        # Find valid corner positions from walls
        corners = self._find_valid_corners()

        for _ in range(count):
            num_routers = random.randint(1, max_routers)

            if corners:
                # Select subset of corners, with some randomization
                selected_corners = random.sample(corners, min(num_routers, len(corners)))
                positions = [(c[0], c[1]) for c in selected_corners]

                # If we need more positions, add random valid ones
                while len(positions) < num_routers:
                    positions.append(self._random_valid_position())
            else:
                positions = [self._random_valid_position() for _ in range(num_routers)]

            router_ids = [random.choice(available_routers).id for _ in range(len(positions))]
            individuals.append(Individual(router_positions=positions, router_ids=router_ids))

        return individuals

    def _create_grid_individuals(
        self,
        count: int,
        available_routers: List[RouterSchema],
        max_routers: int
    ) -> List[Individual]:
        """Create individuals with evenly distributed grid placements."""
        individuals = []

        for _ in range(count):
            num_routers = random.randint(1, max_routers)

            # Calculate grid dimensions for this number of routers
            grid_cols = int(np.ceil(np.sqrt(num_routers)))
            grid_rows = int(np.ceil(num_routers / grid_cols))

            margin = 50
            cell_width = (self.width - 2 * margin) / max(grid_cols, 1)
            cell_height = (self.height - 2 * margin) / max(grid_rows, 1)

            positions = []
            for i in range(num_routers):
                row = i // grid_cols
                col = i % grid_cols

                # Center of each grid cell with small random offset
                x = margin + (col + 0.5) * cell_width + random.uniform(-20, 20)
                y = margin + (row + 0.5) * cell_height + random.uniform(-20, 20)

                # Clamp to valid range
                x = max(margin, min(self.width - margin, x))
                y = max(margin, min(self.height - margin, y))

                # Check if valid, otherwise get random valid position
                ix, iy = int(x), int(y)
                if 0 <= ix < self.width and 0 <= iy < self.height and self.valid_mask[iy, ix]:
                    positions.append((x, y))
                else:
                    positions.append(self._random_valid_position())

            router_ids = [random.choice(available_routers).id for _ in range(len(positions))]
            individuals.append(Individual(router_positions=positions, router_ids=router_ids))

        return individuals

    def _create_centroid_individuals(
        self,
        count: int,
        available_routers: List[RouterSchema],
        max_routers: int
    ) -> List[Individual]:
        """Create individuals with routers at room centroids (if rooms detected)."""
        individuals = []

        # Calculate room centroids
        centroids = []
        if self.map_data.rooms:
            for room in self.map_data.rooms:
                if room.polygon and len(room.polygon) >= 3:
                    cx = sum(p[0] for p in room.polygon) / len(room.polygon)
                    cy = sum(p[1] for p in room.polygon) / len(room.polygon)

                    # Check if centroid is valid
                    ix, iy = int(cx), int(cy)
                    if 0 <= ix < self.width and 0 <= iy < self.height and self.valid_mask[iy, ix]:
                        centroids.append((cx, cy))

        for _ in range(count):
            num_routers = random.randint(1, max_routers)

            if centroids:
                # Select subset of centroids
                selected = random.sample(centroids, min(num_routers, len(centroids)))
                positions = list(selected)

                # If we need more, add random valid positions
                while len(positions) < num_routers:
                    positions.append(self._random_valid_position())
            else:
                # No rooms detected, fall back to random
                positions = [self._random_valid_position() for _ in range(num_routers)]

            router_ids = [random.choice(available_routers).id for _ in range(len(positions))]
            individuals.append(Individual(router_positions=positions, router_ids=router_ids))

        return individuals

    def _create_random_individuals(
        self,
        count: int,
        available_routers: List[RouterSchema],
        max_routers: int
    ) -> List[Individual]:
        """Create purely random individuals for diversity."""
        individuals = []

        for _ in range(count):
            num_routers = random.randint(1, max_routers)
            positions = [self._random_valid_position() for _ in range(num_routers)]
            router_ids = [random.choice(available_routers).id for _ in range(num_routers)]
            individuals.append(Individual(router_positions=positions, router_ids=router_ids))

        return individuals

    def _find_valid_corners(self) -> List[Tuple[float, float]]:
        """Find valid corner positions from wall intersections."""
        corners = []

        if not self.map_data.walls:
            return corners

        # Find wall endpoint clusters (potential corners)
        endpoints = []
        for wall in self.map_data.walls:
            endpoints.append((wall.start.x, wall.start.y))
            endpoints.append((wall.end.x, wall.end.y))

        if not endpoints:
            return corners

        # Cluster nearby endpoints (within 30 pixels)
        clustered = []
        used = set()

        for i, ep1 in enumerate(endpoints):
            if i in used:
                continue

            cluster = [ep1]
            used.add(i)

            for j, ep2 in enumerate(endpoints):
                if j in used:
                    continue
                dist = np.sqrt((ep1[0] - ep2[0])**2 + (ep1[1] - ep2[1])**2)
                if dist < 30:
                    cluster.append(ep2)
                    used.add(j)

            # Average cluster position
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            clustered.append((cx, cy))

        # Filter to valid positions (offset from actual corners)
        for cx, cy in clustered:
            # Try positions near the corner but inside the room
            offsets = [(40, 40), (40, -40), (-40, 40), (-40, -40)]
            for dx, dy in offsets:
                x, y = cx + dx, cy + dy
                ix, iy = int(x), int(y)
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    if self.valid_mask[iy, ix]:
                        corners.append((x, y))
                        break

        return corners

    def _evaluate_fitness(self, individual: Individual) -> float:
        """
        Calculate fitness score for an individual.

        Fitness = weighted combination of:
        - Coverage percentage (maximize)
        - Cost (minimize)
        - Signal uniformity (maximize)
        - Number of routers (minimize)
        - Priority zone coverage (maximize)
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

        # Priority zone coverage score
        priority_score = self._calculate_priority_zone_score(signal_grid, grid_res)

        # Weighted fitness
        if self.constraints.prioritize_cost:
            # Cost-focused weights
            fitness = (
                0.35 * (coverage / 100.0) +
                0.30 * cost_ratio +
                0.15 * uniformity +
                0.10 * router_penalty +
                0.10 * priority_score
            )
        else:
            # Coverage-focused weights (default)
            fitness = (
                0.40 * (coverage / 100.0) +
                0.15 * cost_ratio +
                0.20 * uniformity +
                0.10 * router_penalty +
                0.15 * priority_score
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

    def _calculate_priority_zone_score(
        self,
        signal_grid: SignalGrid,
        grid_res: int
    ) -> float:
        """
        Calculate weighted coverage score for priority zones.

        Returns a score between 0 and 1 based on how well priority zones are covered.
        Higher priority zones contribute more to the score.
        """
        priority_zones = getattr(self.map_data, 'priority_zones', None)
        if not priority_zones:
            return 1.0  # No priority zones defined, return full score

        total_weighted_score = 0.0
        total_weight = 0.0

        for zone in priority_zones:
            if not zone.polygon or len(zone.polygon) < 3:
                continue

            # Use zone-specific min signal or fall back to constraint
            min_signal = zone.min_signal_dbm or self.constraints.min_signal_strength_dbm

            # Count covered and total points in zone
            covered_count = 0
            total_count = 0
            signal_sum = 0.0

            # Sample points within the zone polygon
            # Convert polygon to points for point-in-polygon testing
            polygon_points = np.array(zone.polygon, dtype=np.float32)

            # Get bounding box of zone
            min_x = max(0, int(np.min(polygon_points[:, 0])))
            max_x = min(self.width, int(np.max(polygon_points[:, 0])))
            min_y = max(0, int(np.min(polygon_points[:, 1])))
            max_y = min(self.height, int(np.max(polygon_points[:, 1])))

            # Sample at grid resolution
            for y in range(min_y, max_y, grid_res):
                for x in range(min_x, max_x, grid_res):
                    # Check if point is inside polygon
                    if self._point_in_polygon(x, y, polygon_points):
                        total_count += 1

                        # Get signal at this point from grid
                        grid_y = y // grid_res
                        grid_x = x // grid_res

                        if (0 <= grid_y < signal_grid.grid.shape[0] and
                            0 <= grid_x < signal_grid.grid.shape[1]):
                            signal = signal_grid.grid[grid_y, grid_x]
                            signal_sum += signal

                            if signal >= min_signal:
                                covered_count += 1

            # Calculate zone coverage
            if total_count > 0:
                zone_coverage = covered_count / total_count
                avg_signal = signal_sum / total_count

                # Weight by zone priority
                # Higher priority zones have more impact on the score
                weight = zone.priority

                # Score includes coverage and signal quality
                # Bonus for exceeding min signal threshold
                signal_bonus = max(0, (avg_signal - min_signal) / 20.0)  # Normalize to ~0-1
                zone_score = zone_coverage + 0.2 * min(signal_bonus, 0.5)
                zone_score = min(zone_score, 1.0)

                total_weighted_score += zone_score * weight
                total_weight += weight

        if total_weight == 0:
            return 1.0

        return total_weighted_score / total_weight

    def _point_in_polygon(
        self,
        x: float,
        y: float,
        polygon: np.ndarray
    ) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.
        """
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def _local_search(
        self,
        individual: Individual,
        max_iterations: int = 50
    ) -> Individual:
        """
        Refine solution using hill climbing local search.

        Moves each router in 8 directions with decreasing step sizes,
        keeping improvements and rejecting worse positions.
        """
        best = individual.copy()
        self._evaluate_fitness(best)

        # Step sizes for progressive refinement (pixels)
        step_sizes = [20, 10, 5, 2]

        # 8 directions (N, NE, E, SE, S, SW, W, NW)
        directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]

        for step_size in step_sizes:
            improved = True
            iterations = 0

            while improved and iterations < max_iterations:
                improved = False
                iterations += 1

                # Try to improve each router position
                for router_idx in range(len(best.router_positions)):
                    current_pos = best.router_positions[router_idx]
                    best_pos = current_pos
                    best_local_fitness = best.fitness

                    # Try each direction
                    for dx, dy in directions:
                        new_x = current_pos[0] + dx * step_size
                        new_y = current_pos[1] + dy * step_size

                        # Check bounds
                        if not (10 <= new_x < self.width - 10 and
                                10 <= new_y < self.height - 10):
                            continue

                        # Check if valid position
                        ix, iy = int(new_x), int(new_y)
                        if not (0 <= ix < self.width and 0 <= iy < self.height):
                            continue
                        if not self.valid_mask[iy, ix]:
                            continue

                        # Create candidate solution
                        candidate = best.copy()
                        candidate.router_positions[router_idx] = (new_x, new_y)
                        self._evaluate_fitness(candidate)

                        # Keep if better
                        if candidate.fitness > best_local_fitness:
                            best_pos = (new_x, new_y)
                            best_local_fitness = candidate.fitness

                    # Apply best move for this router
                    if best_pos != current_pos:
                        best.router_positions[router_idx] = best_pos
                        self._evaluate_fitness(best)
                        improved = True

        return best

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
            # Move random router to nearby valid position
            idx = random.randint(0, len(individual.router_positions) - 1)
            old_x, old_y = individual.router_positions[idx]

            # Try to move within a radius, checking for valid positions
            radius = min(self.width, self.height) * 0.2

            # Try multiple times to find valid position
            for _ in range(20):
                new_x = old_x + random.uniform(-radius, radius)
                new_y = old_y + random.uniform(-radius, radius)

                # Clamp to bounds
                new_x = max(10, min(self.width - 10, new_x))
                new_y = max(10, min(self.height - 10, new_y))

                # Check if position is valid
                ix, iy = int(new_x), int(new_y)
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    if self.valid_mask[iy, ix]:
                        individual.router_positions[idx] = (new_x, new_y)
                        break
            else:
                # Couldn't find valid nearby position, get random valid one
                individual.router_positions[idx] = self._random_valid_position()

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
        elif self.map_data.walls:
            # Create boundary polygon from wall endpoints using convex hull
            # This ensures routers are placed inside the floor plan boundary
            all_points = []
            for wall in self.map_data.walls:
                all_points.append([int(wall.start.x), int(wall.start.y)])
                all_points.append([int(wall.end.x), int(wall.end.y)])

            if len(all_points) >= 3:
                points_array = np.array(all_points, dtype=np.int32)

                # Use convex hull to get outer boundary
                hull = cv2.convexHull(points_array)

                # Fill the hull as valid area
                mask_uint8 = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.fillPoly(mask_uint8, [hull], 1)
                mask = mask_uint8.astype(bool)

                # Shrink the valid area by a margin to keep routers away from walls
                kernel = np.ones((60, 60), np.uint8)
                mask_uint8 = mask.astype(np.uint8)
                mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
                mask = mask_uint8.astype(bool)
        else:
            # No walls detected, use center region
            margin = int(min(self.width, self.height) * 0.2)
            mask[margin:self.height-margin, margin:self.width-margin] = True

        # Mark walls as invalid (with buffer)
        for wall in self.map_data.walls:
            x1, y1 = int(wall.start.x), int(wall.start.y)
            x2, y2 = int(wall.end.x), int(wall.end.y)

            # Draw thick line as invalid area (wall buffer)
            mask_uint8 = mask.astype(np.uint8)
            cv2.line(mask_uint8, (x1, y1), (x2, y2), 0, thickness=40)
            mask = mask_uint8.astype(bool)

        # Mark forbidden zones as invalid
        for zone in self.map_data.forbidden_zones:
            if zone.polygon and len(zone.polygon) >= 3:
                pts = np.array([[int(p[0]), int(p[1])] for p in zone.polygon], dtype=np.int32)
                mask_uint8 = mask.astype(np.uint8)
                cv2.fillPoly(mask_uint8, [pts], 0)
                mask = mask_uint8.astype(bool)

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
