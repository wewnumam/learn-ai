import random

# --- Constants and Configuration ---
POPULATION_SIZE = 100  # Number of individuals in the population
GENOME_LENGTH = 50     # Length of the binary string (the "chromosome")
MUTATION_RATE = 0.01   # Probability of a bit flip during mutation
CROSSOVER_RATE = 0.7   # Probability of performing crossover
MAX_GENERATIONS = 100  # Maximum number of generations to run the algorithm
TOURNAMENT_SIZE = 5    # Size of the tournament for parent selection

# --- Core Genetic Algorithm Functions ---

def create_individual(length):
    """
    Creates a single random individual (a binary string).
    An individual's genome is represented as a list of 0s and 1s.
    """
    return [random.randint(0, 1) for _ in range(length)]

def calculate_fitness(individual):
    """
    Calculates the fitness of an individual for the One-Max problem.
    The fitness is simply the number of 1s in its genome.
    """
    return sum(individual)

def tournament_selection(population, fitnesses):
    """
    Selects a parent from the population using tournament selection.
    
    A random subset of the population is chosen (the "tournament"), and the
    fittest individual from that subset is selected as the parent.
    """
    # Select random individuals for the tournament
    tournament_contenders_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    
    # Find the best individual in the tournament
    best_contender_index = -1
    best_fitness = -1
    
    for index in tournament_contenders_indices:
        if fitnesses[index] > best_fitness:
            best_fitness = fitnesses[index]
            best_contender_index = index
            
    return population[best_contender_index]

def single_point_crossover(parent1, parent2):
    """
    Performs single-point crossover on two parents to create two children.
    
    A random crossover point is chosen. The first child gets the first part of
    parent1's genome and the second part of parent2's. The second child gets
    the inverse.
    """
    # Decide if crossover should happen based on the crossover rate
    if random.random() < CROSSOVER_RATE and len(parent1) > 1:
        # Choose a random crossover point (ensuring it's not at the very end)
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        # If no crossover, the children are clones of the parents
        return parent1[:], parent2[:]

def mutate(individual):
    """
    Performs bit-flip mutation on an individual.
    Each bit in the individual's genome has a small chance to be flipped.
    """
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            # Flip the bit (0 becomes 1, 1 becomes 0)
            individual[i] = 1 - individual[i]
    return individual

# --- Main Algorithm Execution ---

def run_genetic_algorithm():
    """
    The main function to execute the genetic algorithm.
    """
    # 1. Initialization: Create the initial population
    population = [create_individual(GENOME_LENGTH) for _ in range(POPULATION_SIZE)]

    print("--- Genetic Algorithm Starting ---")
    print(f"Goal: Find a string of {GENOME_LENGTH} ones.")
    print(f"Population Size: {POPULATION_SIZE}, Max Generations: {MAX_GENERATIONS}\n")

    # Main evolutionary loop
    for generation in range(MAX_GENERATIONS):
        # 2. Fitness Calculation
        fitnesses = [calculate_fitness(ind) for ind in population]
        
        # Find and display the best individual of the current generation
        best_fitness = max(fitnesses)
        best_individual_index = fitnesses.index(best_fitness)
        best_individual = population[best_individual_index]

        print(f"Generation {generation:3}: Best Fitness = {best_fitness}/{GENOME_LENGTH}")

        # 3. Termination Condition Check
        if best_fitness == GENOME_LENGTH:
            print("\n--- Optimal Solution Found! ---")
            print(f"Solution: {''.join(map(str, best_individual))}")
            print(f"Found in generation {generation}")
            return

        # 4. Create the next generation
        next_population = []
        # Elitism: Keep the best individual from the current generation
        next_population.append(best_individual)

        # Generate the rest of the new population through selection, crossover, and mutation
        while len(next_population) < POPULATION_SIZE:
            # Selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Crossover
            child1, child2 = single_point_crossover(parent1, parent2)
            
            # Mutation
            mutate(child1)
            mutate(child2)
            
            # Add new children to the next population
            next_population.append(child1)
            # Ensure population size isn't exceeded if it's an odd number
            if len(next_population) < POPULATION_SIZE:
                next_population.append(child2)

        # Replace the old population with the new one
        population = next_population

    print("\n--- Algorithm Finished ---")
    print("Maximum generations reached without finding the optimal solution.")
    final_best_fitness = max(calculate_fitness(ind) for ind in population)
    print(f"Best fitness achieved: {final_best_fitness}/{GENOME_LENGTH}")

# --- Script Entry Point ---
if __name__ == "__main__":
    run_genetic_algorithm()
