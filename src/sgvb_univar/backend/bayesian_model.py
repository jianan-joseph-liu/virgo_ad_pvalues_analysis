from typing import Tuple, List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
#from ..logging import logger
from .analysis_data import AnalysisData
from .compute_psd import compute_psd

tfd = tfp.distributions
tfb = tfp.bijectors


class BayesianModel:
    def __init__(
            self,
            data: AnalysisData,
            degree_fluctuate: float = None,
            init_params: List[tf.Variable] = None,
    ):

        self.data = data

        if degree_fluctuate is None:
            degree_fluctuate = data.N_delta / 2

        # Setup hyperparams
        self.degree_fluctuate = tf.convert_to_tensor(degree_fluctuate, dtype=tf.float32)
        self.tau0 = tf.convert_to_tensor(0.01, dtype=tf.float32)
        self.c2 = tf.convert_to_tensor(4, dtype=tf.float32)
        self.sig2_alp = tf.convert_to_tensor(10, dtype=tf.float32)
        self.hyper = [self.tau0, self.c2, self.sig2_alp, self.degree_fluctuate]

        self.log_map_vals = tf.Variable(0.0)

        # convert required objects to tensors
        self.trainable_vars = self._get_trainable_vars()
        if init_params is not None:
            for i, p in enumerate(init_params):
                self.trainable_vars[i].assign(p)

        # Initialize model with MAP
        #logger.debug(f"Initialized model with {self.trainable_vars}")

    def _get_trainable_vars(self, batch_size: int = 1) -> List[tf.Variable]:
        #
        #
        # rule:  self.trainable_vars[0, 2, 4] must be corresponding spline regression parameters for p>1
        # in 1-d case, self.trainable_vars[0] must be ga_delta parameters, no ga_theta included.

        # initial values are quite important for training

        p = int(self.data.y_ft.shape[-1])
        size_delta = int(self.data.Xmat_delta.shape[1])
        size_theta = int(self.data.Xmat_theta.shape[1])

        # initializer = tf.initializers.GlorotUniform() # xavier initializer
        # initializer = tf.initializers.RandomUniform(minval=-0.5, maxval=0.5)
        # initializer = tf.initializers.zeros()

        # better to have deterministic inital on reg coef to control
        ga_initializer = tf.initializers.zeros()
        ga_initializer_para = tf.initializers.constant(value=0.0)
        if size_delta <= 10:
            cvec_d = 0.0
        else:
            cvec_d = tf.concat(
                [tf.zeros(10 - 2) + 0.0, tf.zeros(size_delta - 10) + 1.0], 0
            )

        ga_delta = tf.Variable(
            ga_initializer_para(
                shape=(batch_size, p, size_delta), dtype=tf.float32
            ),
            name="ga_delta",
            trainable=True, dtype=tf.float32
        )
        lla_delta = tf.Variable(
            ga_initializer(
                shape=(batch_size, p, size_theta - 2), dtype=tf.float32
            )
            - cvec_d,
            name="lla_delta",
            trainable=True, dtype=tf.float32
        )
        ltau = tf.Variable(
            ga_initializer(shape=(batch_size, p, 1), dtype=tf.float32) - 1,
            name="ltau",
            trainable=True, dtype=tf.float32
        )


        # params:          self.trainable_vars (ga_delta, lla_delta,
        #                                       ltau)
        return [
            ga_delta,
            lla_delta,
            ltau
        ]

    def loglik(self, params: List[tf.Variable]) -> tf.float32:
        # y_re:            self.y_re
        # y_im:            self.y_im
        # Z_:              self.Zar
        # X_:              self.Xmat
        # params:          self.trainable_vars (ga_delta,
        #                                       lla_delta,
        #                                       ltau)
        # each of params is a 3-d tensor with sample_size as the fist dim.
        # self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters


        xγ = tf.matmul(self.data.Xmat_delta, tf.transpose(params[0], [0, 2, 1]))
        sum_xγ = -tf.reduce_sum(xγ, [1, 2])
        exp_xγ_inv = tf.exp(-xγ)


        numerator = tf.square(self.data.y_re) + tf.square(self.data.y_re)
        internal = tf.multiply(numerator, exp_xγ_inv)
        tmp2_ = - tf.reduce_sum(internal, [-2, -1]) # sum over p_dim and freq
        log_lik = tf.reduce_sum(sum_xγ + tmp2_) # sum over all LnL
        return log_lik

    def logpost(self, params: List[tf.Variable]) -> tf.float32:
        return self.loglik(params) + self.logprior(params)

    #
    # Model training one step
    #
    def map_train_step(self, optimizer: Adam) -> tf.float32:  # one training step to get close to MAP
        with tf.GradientTape() as tape:
            self.log_map_vals = -1 * self.logpost(self.trainable_vars)

        grads = tape.gradient(self.log_map_vals, self.trainable_vars)
        grads_and_vars = [(g, v) for g, v in zip(grads, self.trainable_vars) if g is not None]
        if grads_and_vars:
            optimizer.apply_gradients(grads_and_vars)

        self.log_map_vals *= -1  # return POSITIVE log posterior
        return self.log_map_vals

    # For new prior strategy, need new createModelVariables() and logprior()

    def logprior(self, params):
        # hyper:           list of hyperparameters (tau0, c2, sig2_alp, degree_fluctuate)
        # params:          self.trainable_vars (ga_delta, lla_delta,
        #                                       ltau)
        # each of params is a 3-d tensor with sample_size as the fist dim.
        # self.trainable_vars[:,[0, 2, 4]] must be corresponding spline regression parameters
        Sigma1 = tf.multiply(
            tf.eye(tf.constant(2), dtype=tf.float32), self.hyper[2]
        )
        priorDist1 = tfd.MultivariateNormalTriL(
            scale_tril=tf.linalg.cholesky(Sigma1)
        )  # can also use tfd.MultivariateNormalDiag

        Sigm = tfb.Sigmoid()
        s_la_alp = Sigm(
            -tf.range(1, params[1].shape[-1] + 1.0, dtype=tf.float32)
            + self.hyper[3]
        )
        priorDist_la_alp = tfd.HalfCauchy(tf.constant(0, tf.float32), s_la_alp)


        a2 = tf.square(tf.exp(params[1]))
        Sigma2i_diag = tf.divide(
            tf.multiply(
                tf.multiply(a2, tf.square(tf.exp(params[2]))), self.hyper[1]
            ),
            tf.multiply(a2, tf.square(tf.exp(params[2]))) + self.hyper[1],
        )

        priorDist2 = tfd.MultivariateNormalDiag(scale_diag=Sigma2i_diag)

        lpriorAlp_delt = tf.reduce_sum(
            priorDist1.log_prob(params[0][:, :, 0:2]), [1]
        )  #
        lprior_delt = tf.reduce_sum(
            priorDist2.log_prob(params[0][:, :, 2:]), [1]
        )  # only 2 dim due to log_prob rm the event_shape dim
        lpriorla_delt = tf.reduce_sum(
            priorDist_la_alp.log_prob(tf.exp(params[1])), [1, 2]
        ) + tf.reduce_sum(params[1], [1, 2])
        lpriorDel = lprior_delt + lpriorla_delt + lpriorAlp_delt


        priorDist_tau = tfd.HalfCauchy(
            tf.constant(0, tf.float32), self.hyper[0]
        )
        logPrior = (
                lpriorDel
                + tf.reduce_sum(
            priorDist_tau.log_prob(tf.exp(params[2])) + params[2], [1, 2]
        )
        )
        return logPrior

    def compute_psd(
            self,
            vi_samples: List[tf.Tensor],
            quantiles=[0.05, 0.5, 0.95],
            psd_scaling=1.0,
            fs=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return compute_psd(
            self.data.Xmat_delta,
            self.data.Xmat_theta,
            self.data.p,
            vi_samples,
            quantiles,
            psd_scaling,
            fs,
        )
