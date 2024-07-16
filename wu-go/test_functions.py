import numpy as np

class rosenbrock():
    def __init__(self, dim=20):
        self.dim = dim
        self.domain = np.array([dim*[-2, 2]]).reshape(dim, 2)
        self.name = 'rosenbrock'+str(dim)
        self.glob_min = np.ones(dim)

    def __call__(self, conds):
        z = np.sum(np.array([100 * (conds[:, i+1] - conds[:, i] ** 2)**2 + (1 - conds[:, i])**2 for i in range(self.dim - 1)]), axis=0).squeeze()
        return np.log(z + 1e-8)


class tang():
    def __init__(self, dim=20):
        self.dim = dim
        self.domain = np.array([dim*[-5, 5]]).reshape(dim, 2)
        self.name = 'tang'
        self.glob_min = np.ones(dim) * -2.903534

    def __call__(self, conds): # conds x d
        z = np.square(np.square(conds))
        z -= 16*np.square(conds)
        z += 5*conds + 39.16617*self.dim
        z = np.sum(z, axis=1).squeeze()
        return np.log(z)


class ackley():
    def __init__(self):
        self.dim = 2
        self.domain = np.array([[-5, 5], [-5, 5]])
        self.name = 'ackley'
        self.glob_min = np.array([[0, 0]])

    def __call__(self, conds):
        x, y = conds[:, 0], conds[:, 1]
        z = np.log(-20 * np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) -  np.exp(0.5*(np.cos(2*np.pi**x) + np.cos(2*np.pi*y))) + np.e + 20 + 1e-8)
        return z


class levi():
    def __init__(self):
        self.dim = 2
        self.domain = np.array([[-4, 6], [-4, 6]])
        self.name = 'levi'
        self.glob_min = np.array([[1, 1]])

    def __call__(self, conds):
        x, y = conds[:, 0], conds[:, 1]
        z = np.log((np.sin(3*np.pi*x))**2 + ((x - 1)**2) * (1 + (np.sin(3*np.pi*y))**2) + ((y - 1)**2) * (1 + (np.sin(2*np.pi*y))**2) + 1e-8)
        return z


class himmelblau():
    def __init__(self):
        self.dim = 2
        self.domain = np.array([[-5, 5], [-5, 5]])
        self.name = 'himmelblau'
        self.glob_min = np.array([[3.0, 2.0], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]])

    def __call__(self, conds):
        x, y = conds[:, 0], conds[:, 1]
        z = np.log((x**2 + y - 11)**2 + (x + y**2 - 7)**2 + 1e-8)
        return z


class three_hump_camel():
    def __init__(self):
        self.dim = 2
        self.domain = np.array([[-5, 5], [-5, 5]])
        self.name = 'three_hump_camel'
        self.glob_min = np.array([[0, 0]])

    def __call__(self, conds):
        x, y = conds[:, 0], conds[:, 1]
        z = np.log(2 * x**2 - 1.05 * x**4 + x**6 / 6 + x*y + y**2 + 1e-8)
        return z


class holder():
    def __init__(self):
        self.dim = 2
        self.domain = np.array([[-10, 10], [-10, 10]])
        self.name = 'holder'
        self.glob_min = np.array([[8.05502, 9.66459], [-8.05502, -9.66459], [-8.05502, 9.66459], [8.05502, -9.66459]])

    def __call__(self, conds):
        x, y = conds[:, 0], conds[:, 1]
        z = np.log(-np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2)/np.pi))) + 19.2085 + 1e-8)
        return z
