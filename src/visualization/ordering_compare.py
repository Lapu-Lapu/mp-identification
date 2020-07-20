#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:55:39 2019

@author: dom

Bayesian test for ordering hypotheses about repeated draws from several
Bernoulli processes.

Example:
    K many data generating models are tested agaist human observers, who have
    to judge if the data produced by the models look natural or not. The
    responses for each model are assumed to be drawn from a Bernoulli
    distribution with success rate p_k.
    This test evaluates if p_0>=p_1>=....>=p_k for all orderings of the models,
    assuming a uniform prior over the orderings.
    Comparisons involving subsets of the models can be obtained by
    marginalization. Multiple comparsions do not need to be corrected further
    and are guaranteed to be consistent.
"""

import numpy as np
import itertools
import unittest
import scipy.special as spf
import scipy.integrate as igr
from scipy.stats import beta

def log_cumsum(p):
    """Cumsum in the log domain. Returns log(p.cumsum())"""
    max_p = p.max()
    cp = (p - max_p)
    cp = np.log(np.exp(cp).cumsum().clip(1e-200))
    return cp + max_p


def integrate_log_p_joint(counts, h):
    """Integrate Beta-Bernoulli joint probability of data and prior for
    observed counts given that count-generating probabilities are ordered like
    the counts, i.e.
    p_0>=p_1>=...p_{K-1}.
    Priors are assumed to be Beta(1,1) for each process
    h: stepsize of integration
    counts[k,0]: number of time that process k produced a success
    counts[k,1]: number of times that process k produced a failure
    returns: log p(counts|ordering)"""

    # because the integrand for dp_k depends only on p_{k+1}, we can use a
    # sum-product argument and push in the integrals.  We therefore first
    # integrate p_{K-1} and iterate until we reach p_0
    log_pvals = np.log(np.arange(0.0, 1.0 + h / 2, h).clip(1e-100))
    integrals = np.zeros_like(log_pvals)
    log_pvals[0] = -1e100  # log of 0
    log_one_minus_pvals = np.log(
        (1.0 - np.arange(0.0, 1.0 + h / 2, h)).clip(1e-100))
    log_one_minus_pvals[1] = -1e100  # log of 0
    log_h = np.log(h)
    for k in range(len(counts) - 1, -1, -1):
        # log-prob of bernoulli.beta joint density with B(1,1) prior, times
        # previous integrals
        log_prob = (
            counts[k, 0] * log_pvals
            + counts[k, 1] * log_one_minus_pvals
            + integrals
        )

        # trapezoidal rule in the log-domain
        # special case: integral from 0 to 0
        integrals[0] = -1e100
        # special case: integral over [0,h]
        integrals[1] = np.log(0.5) + log_h + np.logaddexp(
            log_prob[0], log_prob[1])
        # rest follows the usual trapezoidal rule
        integrals[2:] = np.logaddexp(
            log_cumsum(log_prob[1:-1]),
            np.log(0.5) + np.logaddexp(log_prob[0], log_prob[2:])) + log_h

    return integrals[-1]


def p_orderings_given_counts(counts, stepsize_factor=0.01):
    """
    Compute log p(ordering of generating probabilities|counts), across all
    possible orderings.
    let K be the number of count-generating processes, assumed to be Bernoulli
    with probabilities p_k.
    counts[k,0]: number of time that process k produced a success
    counts[k,0]: number of times that process k produced a failure
    stepsize_factor: stepsize of numerical integration as a fraction of the
        smallest estimated standard deviation of the beta-posteriors across
        processes. priors are beta(1,1)
    returns: log p(counts|p_0>=p_1>=...>=P_{K-1})"""

    # compute stepsize h from smallest estimated standard deviation across all
    # processes
    beta_stddevs = np.sqrt((counts[:, 0] + 1) * (counts[:, 1] + 1) / (
        (counts[:, 0] + counts[:, 1] + 3.0) *
        (counts[:, 0] + counts[:, 1] + 2)**2))
    h = beta_stddevs.min() * stepsize_factor

    log_posterior = dict()
    marginal_logp = -1e100  # marginal log probability up to prior factor
    for order in itertools.permutations(range(len(counts))):
        ordered_counts = counts[order, :]
        log_posterior[order] = integrate_log_p_joint(ordered_counts, h)
        marginal_logp = np.logaddexp(marginal_logp, log_posterior[order])

    for o in log_posterior.keys():
        log_posterior[o] -= marginal_logp

    return log_posterior


def p_bigger_than_threshold(successes,failures,threshold):
    """Probability that the success probability of a Bernoulli process is greater than threshold,
    given observed successes and failures. Assumes a Beta(1,1) prior"""

    return 1.0-beta.cdf(threshold,successes+1.0,failures+1.0)


class TestOrderingTest(unittest.TestCase):
    def test_log_cumsum(self):
        """Testing log-domain cumulative summation"""
        x = np.random.random(100) * 10.0 + 0.01
        xc = x.cumsum()
        lxc = np.exp(log_cumsum(np.log(x)))
        for a, b in zip(xc, lxc):
            self.assertAlmostEqual(a, b)

    def test_integration_prior(self):
        """Testing prior integration"""
        for num_vars in range(1, 10):
            counts = np.array([[0.0, 0.0]] * num_vars)
            lp = integrate_log_p_joint(counts, 0.0001)
            self.assertAlmostEqual(lp, -spf.gammaln(num_vars + 1), 5)

    def test_integration_3_vars(self):
        """Testing integration against brute-force solution for 3 processes.
        Should work in most cases"""
        counts = np.random.randint(0, 10, size=(3, 2))

        def integrand(p2, p1, p0):
            p = np.array([p0, p1, p2])
            return np.multiply.reduce(
                p**counts[:, 0] * (1.0 - p)**counts[:, 1])

        brute_force, err = igr.tplquad(integrand, 0.0001,
                                       0.9999, lambda x: 0.0, lambda x: x,
                                       lambda x, y: 0.0, lambda x, y: y)
        brute_force = np.log(brute_force)
        sum_prod = integrate_log_p_joint(counts, 0.0001)

        self.assertAlmostEqual(brute_force, sum_prod, 5)

    def test_ordering(self):
        """Testing ordering and normalization"""
        counts = np.array([[5, 5], [7, 3], [12, 2]])

        order_post = p_orderings_given_counts(counts, stepsize_factor=0.1)

        norm = np.exp(np.logaddexp.reduce(list(order_post.values())))

        # check normalization
        self.assertAlmostEqual(norm, 1.0)

        # check most probable ordering
        max_prob = max(list(order_post.values()))
        for o in order_post.keys():
            if order_post[o] == max_prob:
                break

        self.assertTrue(o, (2, 1, 0))


if __name__ == "__main__":

    unittest.main(verbosity=2)

    # usage example:
    # 3 models, with observed counts of successes/failures
    counts = np.array([[5, 5], [5, 3], [12, 2]])

    order_post = p_orderings_given_counts(counts, stepsize_factor=0.1)

    # most probable ordering
    max_prob = max(list(order_post.values()))
    for o in order_post.keys():
        if order_post[o] == max_prob:
            print("Best ordering", o, "with probability", np.exp(max_prob))

    # marginalization examples
    first_idx = 0
    second_idx = 2
    marg = np.exp(
        np.logaddexp.reduce(
            list(
                map(
                    lambda k: order_post[k],
                    filter(
                        lambda i1: i1.index(first_idx) < i1.index(second_idx),
                        order_post.keys())))))
    print("Marginal probability that ", first_idx, ">=", second_idx, ":", marg)

    first_idx = 2
    second_idx = 0
    marg = np.exp(
        np.logaddexp.reduce(
            list(
                map(
                    lambda k: order_post[k],
                    filter(
                        lambda i1: i1.index(first_idx) < i1.index(second_idx),
                        order_post.keys())))))
    print("Marginal probability that 0>=2", first_idx, ">=", second_idx, ":",
          marg)


    # testing if confusion rate is hyperrealistic, i.e. if confusion rate exceeds 0.5
    num_model_chosen=64
    num_natural_chosen=36
    p_hyperreal=p_bigger_than_threshold(num_model_chosen,num_natural_chosen,0.5)
    print("Probability that model is hyperrealistic",p_hyperreal)
