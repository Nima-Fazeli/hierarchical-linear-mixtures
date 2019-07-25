import numpy as np
import pdb


class DummyData():
    def __init__(self):
        self.N = 1000
        # randomly select the split
        self.M = int(np.random.randint(2*self.N/5, 3*self.N/5, 1))

        self.noise = np.array([0.02, 0.01])

        # input
        self.x = np.linspace(-1, 1, num=self.N)
        # linear output
        ## weights drawn from normal distribution
        # self.wl = np.random.normal(size=(2, 2))
        ## weights drawn from uniform distribution
        # self.wl = np.random.rand(2, 2)
        ## weights defined by hand
        self.wl = np.array([[0.1, 0.5],[-0.2, -0.3]])

        self.y = np.empty_like(self.x)
        # polynomial output
        self.par_pol = [-1, 0, 0, 1, 0, 0]
        self.z = np.empty_like(self.x)

    def linear_transform(self, with_shuffle=False):
        if with_shuffle:
            np.random.shuffle(self.x)

        feat_local = np.ones((self.N, 2))
        feat_local[:, 1] = np.reshape(self.x, (self.x.shape[0], -1))[:, 0]
        self.x_out = feat_local

        x1 = self.x_out[:self.M, :]
        x2 = self.x_out[self.M:, :]

        y1 = np.dot(x1, self.wl[0, :]) + np.random.normal(size=x1.shape[0]) * self.noise[0]
        y2 = np.dot(x2, self.wl[1, :]) + np.random.normal(size=x2.shape[0]) * self.noise[1]

        y_noisy = np.concatenate((y1, y2))

        self.y = np.reshape(y_noisy, (self.N, -1)) # converts to Nx1


    def poly_transform(self):
        x1 = self.x[: self.M]
        x2 = self.x[self.M : ]

        z1 = self.par_pol[0] * x1**2 + self.par_pol[1] * x1 + self.par_pol[2]
        z2 = self.par_pol[3] * x2**2 + self.par_pol[4] * x1 + self.par_pol[5]
        z_clean = np.concatenate((z1, z2))

        self.z = z_clean + np.random.normal(size=z_clean.shape[0]) * self.noise
