import numpy as np
import matplotlib.pyplot as plt
# HHO Demonstration: How the algorithm works
# Objective: Sphere function (f(x) = sum(x_i^2))
# Parameters
n_hawks = 20       # Number of hawks (population size)
max_iter = 100     # Number of iterations
dim = 2            # Dimension of problem
lb = -10           # Lower bound
ub = 10            # Upper bound
# Objective Function
def sphere(x):
    return np.sum(x**2)
# Levy flight for rapid dives
def levy_flight(dim, beta=1.5):
    sigma_u = (np.math.gamma(1+beta) * np.sin(np.pi*beta/2) /
               (np.math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u = np.random.randn(dim) * sigma_u
    v = np.random.randn(dim)
    return u / np.abs(v)**(1/beta)
# Initialize hawks
hawks = np.random.uniform(lb, ub, (n_hawks, dim))
fitness = np.apply_along_axis(sphere, 1, hawks)
best_idx = np.argmin(fitness)
prey = hawks[best_idx].copy()
prey_fitness = fitness[best_idx]
convergence = []
# HHO Main Loop
for t in range(max_iter):
    E0 = 2 * (1 - t/max_iter)  # Energy decreases over time

    for i in range(n_hawks):
        E = 2*np.random.rand()*E0 - E0
        r = np.random.rand()
        J = 2*(1-np.random.rand())

        if abs(E) >= 1:
            # Exploration phase
            rand_hawk = hawks[np.random.randint(0, n_hawks)]
            hawks[i] = rand_hawk - np.random.rand() * abs(rand_hawk - 2*np.random.rand()*hawks[i])
        else:
            # Exploitation phase
            if r >= 0.5:
                hawks[i] = prey - E * abs(prey - hawks[i])
            else:
                hawks[i] = prey - E * abs(prey - hawks[i]) + J*levy_flight(dim)

        # Keep within bounds
        hawks[i] = np.clip(hawks[i], lb, ub)
    # Evaluate fitness
    fitness = np.apply_along_axis(sphere, 1, hawks)
    best_idx = np.argmin(fitness)
    if fitness[best_idx] < prey_fitness:
        prey = hawks[best_idx].copy()
        prey_fitness = fitness[best_idx]

    convergence.append(prey_fitness)

# Results
print("HHO Demonstration (Sphere Function)")
print("Best solution:", prey)
print("Best fitness:", prey_fitness)

# Convergence plot
plt.plot(convergence, 'b-', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Fitness (Sphere Value)")
plt.title("HHO Convergence - Demonstration")
plt.grid(True)
plt.show()
