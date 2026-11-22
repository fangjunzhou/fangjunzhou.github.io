from gaussian_pairings import GaussianPairings


class IdenticalGaussianPairings(GaussianPairings):
    generated_std = 2
    convergence_ticks = range(0, 21, 4)
