import numpy as np
from scipy import sparse

class SoftmaxRegression:
    def __init__(self, W_init, eta=0.01, tol=1e-4, max_count=100000):
        self.W = [W_init]  
        self.eta = eta  
        self.tol = tol 
        self.max_count = max_count  

    def convert_labels(self, y, C):
        Y = sparse.coo_matrix((np.ones_like(y), 
            (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
        return Y

    def softmax(self, Z):
        e_Z = np.exp(Z)
        A = e_Z / e_Z.sum(axis = 0)
        return A

    def softmax_stable(self, Z):
        e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
        A = e_Z / e_Z.sum(axis = 0)
        return A

    def fit(self, X, y):
        C = self.W[0].shape[1]
        Y = self.convert_labels(y, C)
        N = X.shape[1]
        d = X.shape[0]
        count = 0
        check_w_after = 20
        
        while count < self.max_count:
            
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = X[:, i].reshape(d, 1)
                yi = Y[:, i].reshape(C, 1)
                ai = self.softmax(np.dot(self.W[-1].T, xi))
                W_new = self.W[-1] + self.eta * xi.dot((yi - ai).T)
                count += 1
                
                if count % check_w_after == 0:  
                    if np.linalg.norm(W_new - self.W[-check_w_after]) < self.tol:
                        return self.W
                self.W.append(W_new)
        return self.W

    def predict(self, X):
        A = self.softmax_stable(np.dot(self.W[-1].T, X))
        return np.argmax(A, axis=0)

