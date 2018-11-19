""" Solve a Laplace problem using finite elements
and verify via the Method of Manufactured Solution.
"""
import firedrake as fe
import math


div, grad, dot = fe.div, fe.grad, fe.dot

sin, pi = fe.sin, fe.pi

dx = fe.dx

def R(u):
    """ Strong form residual """    
    return div(grad(u))

    
def F(u, v):
    """ Weak form residual """
    return -dot(grad(v), grad(u))*dx


def manufactured_solution(mesh):
    
    x = fe.SpatialCoordinate(mesh)[0]
    
    return sin(pi*x)
    
    
def compute_space_accuracy_via_mms(grid_sizes, element_degree):

    print("h, L2_norm_error")
    
    h, e = [], []
    
    for gridsize in grid_sizes:
    
        mesh = fe.UnitIntervalMesh(gridsize)
        
        element = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
        
        V = fe.FunctionSpace(mesh, element)
        
        u_m = manufactured_solution(mesh)
        
        bc = fe.DirichletBC(V, u_m, "on_boundary")
        
        u = fe.TrialFunction(V)
        
        v = fe.TestFunction(V)
        
        u_h = fe.Function(V)
        
        fe.solve(F(u, v) == v*R(u_m)*dx, u_h, bcs = bc)

        e.append(math.sqrt(fe.assemble(fe.inner(u_h - u_m, u_h - u_m)*dx)))
        
        h.append(1./float(gridsize))
        
        print(str(h[-1]) + ", " + str(e[-1]))
    
    r = h[-2]/h[-1] 
    
    log = math.log
    
    order = log(e[-2]/e[-1])/log(r)
    
    return order
    
    
if __name__ == "__main__":

    order = compute_space_accuracy_via_mms(
        grid_sizes = (8, 16), element_degree = 1)
    
    print("Observed spatial order of accuracy is " + str(order))
    