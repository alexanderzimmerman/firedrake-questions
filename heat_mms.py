""" Solve a problem governed by the heat equation using finite elements
and verify via the Method of Manufactured Solution.
"""
import firedrake as fe
import math


diff, div, grad, dot, exp = fe.diff, fe.div, fe.grad, fe.dot, fe.exp

dx = fe.dx

Delta_t = fe.Constant(1.)

t = fe.Constant(0.)

TIME_EPSILON = 1.e-8

def R(u):
    """ Strong form residual """    
    return diff(u, t) - div(grad(u))

    
def F(u, v, un):
    """ Weak form residual """
    return (v*(u - un)/Delta_t + dot(grad(v), grad(u)))*dx
    
    
def manufactured_solution(mesh):
    
    x = fe.SpatialCoordinate(mesh)[0]
    
    sin, pi = fe.sin, fe.pi
    
    return sin(pi*x)*exp(-t)
    
    
def compute_time_accuracy_via_mms(
        gridsize, element_degree, timestep_sizes, endtime):
    
    mesh = fe.UnitIntervalMesh(gridsize)
    
    element = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
    
    V = fe.FunctionSpace(mesh, element)
    
    u_h = fe.Function(V)
    
    v = fe.TestFunction(V)
    
    un = fe.Function(V)
    
    u_m = manufactured_solution(mesh)
    
    bc = fe.DirichletBC(V, u_m, "on_boundary")
    
    _F = F(u_h, v, un)
    
    problem = fe.NonlinearVariationalProblem(
        _F - v*R(u_m)*dx,
        u_h,
        bc,
        fe.derivative(_F, u_h))
    
    solver = fe.NonlinearVariationalSolver(problem)
    
    t.assign(0.)
    
    initial_values = fe.interpolate(u_m, V)
    
    L2_norm_errors = []
    
    print("Delta_t, L2_norm_error")
    
    for timestep_size in timestep_sizes:
        
        Delta_t.assign(timestep_size)
        
        un.assign(initial_values)
        
        time = 0.
        
        t.assign(time)
        
        while time < (endtime - TIME_EPSILON):
            
            time += timestep_size
            
            t.assign(time)
            
            solver.solve()
            
            un.assign(u_h)

        L2_norm_errors.append(
            math.sqrt(fe.assemble(fe.inner(u_h - u_m, u_h - u_m)*dx)))
        
        print(str(timestep_size) + ", " + str(L2_norm_errors[-1]))
    
    r = timestep_sizes[-2]/timestep_sizes[-1]
    
    e = L2_norm_errors
    
    log = math.log
    
    order = log(e[-2]/e[-1])/log(r)
    
    return order
    
    
if __name__ == "__main__":
    
    order = compute_time_accuracy_via_mms(
        gridsize = 16,
        element_degree = 1,
        timestep_sizes = (1./8., 1./16.),
        endtime = 1.)
    
    print("Observed temporal order of accuracy is " + str(order))
    