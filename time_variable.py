import firedrake as fe


mesh = fe.UnitIntervalMesh(2)

x = fe.SpatialCoordinate(mesh)[0]

sin, pi, exp, diff = fe.sin, fe.pi, fe.exp, fe.diff

for t in (fe.Constant(0.), fe.variable(0.)):

    u = sin(pi*x)*exp(-t)

    print(diff(u, t))
