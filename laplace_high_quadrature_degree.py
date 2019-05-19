""" Solve a Laplace problem using finite elements
and verify via the Method of Manufactured Solution.
"""
import firedrake as fe
import math


div, grad, dot = fe.div, fe.grad, fe.dot

tanh, pi = fe.tanh, fe.pi

parameters = {"smoothing": fe.Constant(1.)}

def R(u):
        """ Strong form residual """    
        return div(grad(u))

    
def F(u, v):
        """ Weak form residual """
        return -dot(grad(v), grad(u))
        
        
def manufactured_solution(mesh):
    
    x = fe.SpatialCoordinate(mesh)[0]
    
    s = parameters["smoothing"]
    
    return tanh(pi*x/s)
    
    
def compute_space_accuracy_via_mms(
        grid_sizes, element_degree, quadrature_degree, smoothing):
    
    dx = fe.dx(degree = quadrature_degree)
    
    parameters["smoothing"].assign(smoothing)
    
    h, e, order = [], [], []
    
    for gridsize in grid_sizes:
    
        mesh = fe.UnitIntervalMesh(gridsize)
        
        element = fe.FiniteElement("P", mesh.ufl_cell(), element_degree)
        
        V = fe.FunctionSpace(mesh, element)
        
        u_m = manufactured_solution(mesh)
        
        bc = fe.DirichletBC(V, u_m, "on_boundary")
        
        u = fe.TrialFunction(V)
        
        v = fe.TestFunction(V)
        
        u_h = fe.Function(V)
        
        fe.solve(F(u, v)*dx == v*R(u_m)*dx, u_h, bcs = bc)

        e.append(math.sqrt(fe.assemble(fe.inner(u_h - u_m, u_h - u_m)*dx)))
        
        h.append(1./float(gridsize))
        
        if len(e) > 1:
    
            r = h[-2]/h[-1] 
            
            log = math.log
            
            order = log(e[-2]/e[-1])/log(r)
            
            print("{0: <4}, {1: .3f}, {2: .5f}, {3: .3f}, {4: .3f}".format(
                str(quadrature_degree), smoothing, h[-1], e[-1], order))
            
    
if __name__ == "__main__":

    print("q,     s,      h,        error,  order")
    
    for q in (None, 1):
    
        for s in (1., 0.1, 0.01):
        
            compute_space_accuracy_via_mms(
                grid_sizes = (8, 16, 32, 64, 128, 256), 
                element_degree = 1, 
                quadrature_degree = q, 
                smoothing = s)
            