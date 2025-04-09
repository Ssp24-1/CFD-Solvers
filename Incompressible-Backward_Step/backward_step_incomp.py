import numpy as np
import matplotlib.pyplot as plt

# Fluid properties
density = 1.0                   # Density of the fluid (kg/m^3)
velocity = 1.0                  # Velocity of the fluid (m/s)
reynold_number = 150            # Reynolds number (Re = 150)
domain_size = 1.0

kinematic_viscosity = density * velocity * domain_size / reynold_number           # Kinematic viscosity of the fluid (m^2/s)

# Grid parameters
N = 101                         # Number of grid points in each direction (x and y) (N-1 divisions of the domain)

# Time Parameters
dt = 0.0025                     # Time step
iter = 3000                     # Number of iterations


# ========== #


# Initialize grid
x = np.linspace(0, domain_size, N)
y = np.linspace(0, domain_size, N)
h = domain_size / (N-1)

[X, Y] = np.meshgrid(x, y)          # Creating a mesh grid with the coordinates of each point

u_prev = np.zeros_like(X)
v_prev = np.zeros_like(X)           # Initializing all unknown variables as 0
p_prev = np.zeros_like(X)


# ========== #


# Define all mathematical operators used in NS Equations

# (1) Partial Diff. Operator (Central Difference Scheme)

    # Since we are using central difference scheme, the last rows and columns cannot be calculated since there isnt 
    # any values defined outside the grid. Only the interior points are referenced here. (If required, additional boundary points should be added)

def central_diff_x(f):
    diff = np.zeros_like(f)

    diff[1: -1, 1: -1] = (f[1: -1, 2: ] - f[1: -1, 0: -2]) / (2*h)

    return diff

def central_diff_y(f):
    diff = np.zeros_like(f)

    diff[1: -1, 1: -1] = (f[2: , 1: -1] - f[0: -2, 1: -1]) / (2*h)

    return diff


# (2) Laplace Operator

    # 5 point Stencil is used here to approximate the laplace:
    # ∇**2(u) = [u(i, j+1) + u(i, j-1) + u(i+1, j) + u(i-1, j) - 4u(i, j)] / h**2

def laplace_op(f):
    diff = np.zeros_like(f)

    diff[1: -1, 1: -1] = (f[1: -1, 2: ] + f[1: -1, 0: -2] + f[2: , 1: -1] + f[0: -2, 1: -1] - 4*f[1: -1, 1: -1]) / h**2

    return diff


# ========== #


# Solve NS Equations

    # Projection method is used here since the flow is incompressible, split the velocity and pressure terms and correct veloctity with pressure poisson's equation

for i in range(iter):

    print(i)

    d_u_prev__d_x = central_diff_x(u_prev)
    d_u_prev__d_y = central_diff_y(u_prev)
    d_v_prev__d_x = central_diff_x(v_prev)
    d_v_prev__d_y = central_diff_y(v_prev)
    
    laplace__u_prev = laplace_op(u_prev)
    laplace__v_prev = laplace_op(v_prev)

    # Step (1): Solve momentum equation without pressure gradient

    u_prime = u_prev + dt * ((kinematic_viscosity * laplace__u_prev) - u_prev * d_u_prev__d_x - v_prev * d_u_prev__d_y)

    v_prime = v_prev + dt * ((kinematic_viscosity * laplace__v_prev) - u_prev * d_v_prev__d_x - v_prev * d_v_prev__d_y)

    # Step (1.5): Enforce Velocity Boundary Conditions:
    u_prime[0, :] = 0.0                 # Bottom
    u_prime[:, 0] = velocity          # Left (Inlet)
    u_prime[:, -1] = 0.0                # Right
    u_prime[-1, :] = 0.0                # Top

    v_prime[0, :] = 0.0                 # Bottom
    v_prime[:, 0] = 0.0                 # Left (Inlet)
    v_prime[:, -1] = 0.0                # Right
    v_prime[-1, :] = 0.0                # Top 


    u_prime[0:26, 0:26]  = 0
    v_prime[0:26, 0:26]  = 0
    
    # Step (2): Setup pressure poisson's equation to calculate p 

    # We have the pressure poisson's euqation by taking the divergence of momentum equation and applying incompressible BC
    # ∇**2(p) = (ρ/dt) * (du/dx + dv/dy)
    # Calculate p using the same 5 point stencil approach:
    # p(i, j) = (1/4) * [p(i, j+1) + p(i, j-1) + p(i+1, j) + p(i-1, j) - (∇**2(p) * h**2)] (Substitute laplace of p with previous equation)

    # Convergence was not achieved unless iterations were set very high or never converges, hence we smoothen out the pressure values for each iteration
    # The value is run through an iterative loop for 50 iterations.

    d_u_prime__d_x = central_diff_x(u_prime)
    d_v_prime__d_y = central_diff_y(v_prime)
    
    rhs = (density/dt) * (d_u_prime__d_x + d_v_prime__d_y)

    for i in range(50):
        p_next = np.zeros_like(p_prev)

        p_next[1: -1, 1: -1] = (1/4) * (p_prev[1: -1, 2: ] + p_prev[1: -1, 0: -2] + p_prev[2: , 1: -1] + p_prev[0: -2, 1: -1] - rhs[1: -1, 1: -1]*h**2)
        
        # Enforce Pressure Boundary Condition: Since we are using Neumann BC(Flux = 0), We can equate the value of P adjacant to the boundary to the boundary itself
        p_next[:, -1] = p_next[:, -2]           # Right
        p_next[:, 0] = p_next[:, 1]             # Left
        p_next[0, :] = p_next[1, :]             # Bottom
        p_next[-1, :] = p_next[-2, :]           # Top

        p_next[0:25, 0:25] = 0.5                    # Important to set the values of the pressure inside the step to prevent flow deviations. Set to 0.5 based on freestream dynamic pressure given the BCs.

        p_next[0:26, 0:26] = p_next[27, 0:26]       # Neumann BC on the surfaces of the backward step. (0 -> 25 ; 0 -> 25) is the step coordinates
        p_next[0:26, 0:26] = p_next[0:26, 27]       # Neumann BC on the surfaces of the backward step. (0 -> 25 ; 0 -> 25) is the step coordinates

        p_prev = p_next


    # Step (3): Compute velocity correction:  (Projection method)

    # u = u_prime - (dt/ρ) * (dp/dx)                   
    # v = v_prime - (dt/ρ) * (dp/dy)  

    d_p_next__d_x = central_diff_x(p_next)
    d_p_next__d_y = central_diff_y(p_next)
    
    u_next = u_prime - (dt/density) * d_p_next__d_x
    v_next = v_prime - (dt/density) * d_p_next__d_y

    # Final Enforce of Velocity BC
    u_next[0, :] = 0.0              # Bottom
    u_next[:, 0] = velocity       # Left (Inlet)
    u_next[:, -1] = 0.0             # Right
    u_next[-1, :] = 0.0             # Top (BC)

    v_next[0, :] = 0.0              # Bottom
    v_next[:, 0] = 0.0              # Left (Inlet)
    v_next[:, -1] = 0.0             # Right
    v_next[-1, :] = 0.0             # Top 


    u_prime[0:26, 0:26]  = 0        # Velocity within the step is set to 0.
    v_prime[0:26, 0:26]  = 0        # Velocity within the step is set to 0.



    # Check if velocity and pressure values have stabilized
    if i == 1:
        continue
    else:
        if np.max(np.abs(u_next - u_prev)) < 1e-4 and np.max(np.abs(v_next - v_prev)) < 1e-4 and np.max(np.abs(p_next - p_prev)) < 1e-4:
            print("Converged")
            break
        else:
            print("Not Converged")
    
    u_prev = u_next
    v_prev = v_next
    p_prev = p_next

# Since this is incompressible flow, only pressure gradients make sense, the absolute pressure are valid upto a constant. 
    # To approximate actual pressure values, take the mean of the pressure values and subtract it from the pressure values

mean_p = np.mean(p_next)
p_next = p_next - mean_p


# Step (4): Displace the contours and streamline plots

plt.figure()

plt.contourf(X, Y, p_next)
plt.colorbar()

# Add a black box from 0:25 in X and Y
box_x = 0
box_y = 0
box_width = X[0, 25] - X[0, 0]  # Width of the box
box_height = Y[25, 0] - Y[0, 0]  # Height of the box

rectangle = plt.Rectangle((box_x, box_y), box_width, box_height, color='black', fill=True)
plt.gca().add_patch(rectangle)

plt.streamplot(X, Y, u_next, v_next, color='black')

plt.show()