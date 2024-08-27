import numpy as np


class DenseHopfield:
    def __init__(self, pat_size):
        self.size = pat_size
        self.beta = 1
        self.max_norm = np.sqrt(self.size)

        # if normalization_option == 1:
        #    # normalize dot product of patterns by 1/sqrt(pattern_size)
        #    self.energy = self.energy_normalized

        return

    def store(self, X):
        """expects patterns as NxD numpy array, a row-wise in pattern matrix 
        """
        self.num_pat = len(X)
        assert type(X) is np.ndarray, 'input patterns must be numpy arrays'
        assert len(X.shape) == 2, 'input patterns must have 2 dimensions'

        self.X = np.array(X)  # save patterns row-wise

    def retrieve(self, xis, max_iter=np.inf, thresh=0.5, verbose=False):
        """expects disturbed patterns which are going to be restored as a MxD matrix, a row-wise pattern matrix
        """

        new_xis = []
        for xi in xis:
            new_xis.append(self._retrieve_single(
                xi, max_iter=max_iter, thresh=thresh, verbose=verbose))
        return np.array(new_xis)

    def _retrieve_single(self, xi, max_iter=np.inf, thresh=0.5, verbose=False):
        """expects single disturbed pattern which is going to be restored as a D-dim vector
        """
        if xi.shape[-1] != self.size:
            raise ValueError("Input pattern %r does not match state size: %d vs %d"
                             % (xi, xi.shape[-1], self.size))
        assert type(xi) == np.ndarray, 'input pattern was no numpy array'

        pat_old = xi.copy()
        step = 0

        while(step < max_iter):
            if verbose:
                E = self.energy(xi)
                print(f"step {step}: E={E}")

            pat_new = np.zeros(xi.shape)
            # jj = np.random.randint(self.size)
            for jj in range(self.size):
                # simple variant:
                E = 0
                pat_orig = pat_old[jj].copy()
                pat_old[jj] = +1
                E -= self.energy(pat_old)
                pat_old[jj] = -1
                E += self.energy(pat_old)

                pat_old[jj] = pat_orig
                pat_new[jj] = np.where(E > 0, 1, -1)

            if np.count_nonzero(pat_old != pat_new) <= thresh:
                break
            else:
                pat_old = pat_new

            step += 1

        return pat_new

    def energy(self, X):
        return -1*np.sum(np.exp(self.X @ X))


class ModernHopfield:
    def __init__(self, pat_size, beta=32):
        self.size = pat_size
        self.beta = beta
        self.max_norm = np.sqrt(self.size)

        return

    def store(self, X):
        """expects patterns as NxD numpy array, a row-wise in pattern matrix 
        """
        self.num_pat = X.shape[0]
        assert type(X) is np.ndarray, 'input patterns must be numpy arrays'
        assert len(X.shape) == 2, 'input patterns must have 2 dimensions'

        self.X = np.array(X)  # save patterns row-wise
        # maximal norm of actually stored patterns
        self.M = max(np.linalg.norm(x) for x in X)

    def retrieve(self, xis, max_iter=np.inf, thresh=5e-15, verbose=False):
        """expects disturbed patterns which are going to be restored as a MxD matrix, a row-wise pattern matrix
        """

        new_xis = []
        for xi in xis:
            new_xis.append(self._retrieve_single(
                xi, max_iter=max_iter, thresh=thresh, verbose=verbose))
        return np.array(new_xis)

    def _retrieve_single(self, xi, max_iter=100, thresh=0.5, verbose=False):
        """expects single disturbed pattern which is going to be restored as a D-dim vector
        """
        if xi.shape[-1] != self.size:
            raise ValueError("Input pattern %r does not match state size: %d vs %d"
                             % (xi, xi.shape[-1], self.size))
        assert type(xi) == np.ndarray, 'input pattern was no numpy array'

        pat_old = xi.copy()
        step = 0

        while(step < max_iter):
            if verbose:
                E = self.energy(xi)
                print(f"step {step}: E={E}")

            pat_new = self.X.T @ self.softmax(self.beta * self.X @ pat_old)

            if np.count_nonzero(pat_old != pat_new) <= thresh:
                break
            else:
                pat_old = pat_new

            step += 1

        return pat_new

    @staticmethod
    def softmax(z):  # unnormalized
        maxz = z.max()
        numerators = np.exp(z-maxz)  # top
        denominator = np.sum(numerators)  # bottom
        return numerators/denominator

    @staticmethod
    def _lse(z, beta):

        maxz = z.max()
        return 1/beta * np.log(np.sum(np.exp(beta*(z-maxz))))

    def energy(self, X):
        return -1*self._lse(self.X.T @ X, 1) + 0.5 * X.T @ X\
            + 1/self.beta*np.log(self.num_pat)\
            + 0.5*self.M**2
