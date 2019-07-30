import pymc3 as pm
import numpy as np
from theano import shared, tensor as tt


class MixLinBayes():
    def __init__(self, x_train, y_train):
        """
        args:   x_train - NxF numpy array - N: data count, F: feature count
                y_train - NxD numpy array - N: data count, D: output dimension
        """
        # maximum number of mixtures
        self.K = 20

        # number of models
        self.D = y_train.shape[1]

        # number of features
        if x_train.ndim == 1:
            self.F = 1
        else:
            self.F = x_train.shape[1]

        self.yt = y_train

        if x_train.ndim == 1:
            self.x_shared = shared(x_train[:, np.newaxis], broadcastable=(False, True))
        else:
            self.x_shared = shared(x_train, broadcastable=(False, True))

        self.model = self._models()


    def _models(self):
        model = []

        # loop over the number of models appending to the list
        for i in range(self.D):
            with pm.Model() as mdl:
                alpha = pm.Normal('alpha', 0., 5., shape=self.K)

                # v = norm_cdf(alpha + beta * x_shared)
                # w = pm.Deterministic('w', stick_breaking(v))

                beta = []
                for i in range(self.F):
                    beta.append(pm.Normal('beta{}'.format(i), 0., 5., shape=self.K))

                if self.F == 1:
                    v_med = beta[0] * self.x_shared
                else:
                    v_med = beta[0] * self.x_shared[:, 0]

                for i in range(1, self.F):
                    v_med += beta[i] * self.x_shared[:, i]

                v = self.norm_cdf(alpha + v_med)
                w = pm.Deterministic('w', self.stick_breaking(v))

                # offset
                gamma = pm.Normal('gamma', 0., 10., shape=self.K)
                delta = []
                for i in range(self.F):
                    delta.append(pm.Normal('delta{}'.format(i), 0., 5., shape=self.K))

                if self.F == 1:
                    d_med = delta[0] * self.x_shared
                else:
                    d_med = delta[0] * self.x_shared[:, 0]

                for i in range(1, self.F):
                    d_med += delta[i] * self.x_shared[:, i]

                mu = pm.Deterministic('mu', gamma + d_med)

                tau = pm.Gamma('tau', 1., 1., shape=self.K)
                obs = pm.NormalMixture('obs', w, mu, tau=tau, observed=self.yt[:, i])

            model.append(mdl)

        return model


    def train(self):
        SAMPLES = 20000 # 20000
        BURN = 10000    # 10000
        traces = []
        for i in range(self.D):
            with self.model[i]:
                step = pm.Metropolis()
                trace = pm.sample(SAMPLES, step, chains=1, tune=BURN)
            traces.append(trace)

        self.trace = traces

    def predict(self, x, with_mean = True):
        self.x_shared.set_value(x[:, np.newaxis])

        if x.ndim == 1:
            self.x_shared.set_value(x[:, np.newaxis])
        else:
            self.x_shared.set_value(x)

        PP_SAMPLES = 5000 # 5000
        pp_traces = []
        for i in range(self.D):
            with self.model[i]:
                pp_trace = pm.sample_ppc(self.trace[i], PP_SAMPLES)

            if with_mean:
                pp_traces.append(pp_trace['obs'].mean(axis=0))
            else:
                pp_traces.append(pp_trace)

        return pp_traces


    @staticmethod
    def norm_cdf(z):
        return 0.5 * (1 + tt.erf(z / np.sqrt(2)))

    @staticmethod
    def stick_breaking(v):
        return v * tt.concatenate([tt.ones_like(v[:, :1]),
                                   tt.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                                  axis=1)
