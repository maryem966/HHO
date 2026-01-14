import numpy as np
import matplotlib.pyplot as plt

# HHO Engineering Example: Minimize Fuel Consumption
# Objective: Fuel = speed^2 * engine_param - 50*engine_param + 200
# Variables: speed (40-120 km/h), engine parameter (0.5-2.0)
# Parameters
n_hawks = 15
max_iter = 100
dim = 2
lb = np.array([40, 0.5])   # Lower bounds: [speed, engine param]
ub = np.array([120, 2.0])  # Upper bounds: [speed, engine param]
# Objective Function
def fuel_consumption(x):
    speed, param = x
    fuel = speed**2 * param - 50 * param + 200
    return fuel
# Levy flight
def levy_flight(dim, beta=1.5):
    sigma_u = (np.math.gamma(1+beta) * np.sin(np.pi*beta/2) /
               (np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(dim) * sigma_u
    v = np.random.randn(dim)
    return u / np.abs(v)**(1/beta)
# Initialize hawks
hawks = np.random.uniform(lb, ub, (n_hawks, dim))
fitness = np.apply_along_axis(fuel_consumption, 1, hawks)
best_idx = np.argmin(fitness)
prey = hawks[best_idx].copy()
prey_fitness = fitness[best_idx]
convergence = []
# HHO Main Loop
for t in range(max_iter):
    E0 = 2 * (1 - t/max_iter)

    for i in range(n_hawks):
        E = 2*np.random.rand()*E0 - E0
        r = np.random.rand()
        J = 2*(1-np.random.rand())

        if abs(E) >= 1:
            # Exploration
            rand_hawk = hawks[np.random.randint(0, n_hawks)]
            hawks[i] = rand_hawk - np.random.rand() * abs(rand_hawk -
            2*np.random.rand()*hawks[i])
        else:
            # Exploitation
            if r >= 0.5:
                hawks[i] = prey - E * abs(prey - hawks[i])
            else:
                hawks[i] = prey - E * abs(prey - hawks[i]) + J*levy_flight(dim)

        # Boundary check
        hawks[i] = np.clip(hawks[i], lb, ub)
    # Evaluate fitness
    fitness = np.apply_along_axis(fuel_consumption, 1, hawks)
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < prey_fitness:
        prey = hawks[best_idx].copy()
        prey_fitness = fitness[best_idx]

    convergence.append(prey_fitness)
# Results
print("HHO Engineering Example - Fuel Optimization")
print("Optimal speed and engine parameter:", prey)
print("Minimum fuel consumption:", prey_fitness)
# Convergence plot
plt.plot(convergence, 'b-', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Fuel Consumption")
plt.title("HHO Convergence - Fuel Optimization")
plt.grid(True)
plt.show()
