import firedrake as fe
import matplotlib
import matplotlib.pyplot as pp


mesh = fe.UnitIntervalMesh(8)

element = fe.FiniteElement("P", mesh.ufl_cell(), 1)

V = fe.FunctionSpace(mesh, element)


x = fe.SpatialCoordinate(mesh)[0]

f = x

g = 2.*x


fig = pp.figure()

axes = pp.axes()

fe.plot(fe.interpolate(f, V), axes = axes, color = "red")

fe.plot(fe.interpolate(g, V), axes = axes, color = "blue")

pp.legend((r"$f$", r"$g$"))

pp.xlabel(r"$x$")

pp.savefig("plot_line_color.png")

pp.show()
