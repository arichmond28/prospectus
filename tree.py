import numpy as np

class TreeNode(object):  
    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction

def sqsplit(xTr,yTr,weights=[]):
    N,D = xTr.shape
    assert D > 0 # must have at least one dimension
    assert N > 1 # must have at least two samples
    if isinstance(weights, list) and weights == []: # if no weights are passed on, assign uniform weights
        weights = np.ones(N)
    elif weights.size == 0:
        weights = np.ones(N)
    weights = weights/sum(weights) # Weights need to sum to one (we just normalize them)
    bestloss = np.inf
    feature = np.inf
    cut = np.inf
    for d in range(D):
        ii = xTr[:,d].argsort() # sort data along the dth dimension
        xs = xTr[ii,d] # sorted feature values
        ws = weights[ii] # sorted weights
        ys = yTr[ii] # sorted labels
        
        # Initialize right side quantities
        W_R = np.sum(ws)  # total weight on the right side
        P_R = np.sum(ws * ys)  # weighted sum of labels on the right side
        Q_R = np.sum(ws * ys**2)  # weighted sum of labels squared on the right side
        
        # Initialize left side quantities
        W_L = 0.0
        P_L = 0.0
        Q_L = 0.0
        
        # np.finfo(float).eps gives us the smallest possible positive number that can be represented by floats. 
        idif = np.where(np.abs(np.diff(xs, axis=0)) > np.finfo(float).eps * 100)[0]
        pj = 0

        for j in idif:
            # Todo: 
            for i in range(pj, j+1):
                W_L += ws[pj]
                P_L += ws[pj] * ys[pj]
                Q_L += ws[pj] * ys[pj]**2
                W_R -= ws[pj]
                P_R -= ws[pj] * ys[pj]
                Q_R -= ws[pj] * ys[pj]**2
                if i+1<N and xs[i] != xs[i + 1] + W_L != 0 and W_R != 0:
                    loss = (Q_R-(P_R**2)/W_R) + (Q_L-(P_L**2)/W_L)
                    if (loss < bestloss):
                        feature = d
                        cut = (xs[pj] + xs[pj + 1]) / 2
                        bestloss = loss
            pj = j + 1
            
    assert feature != np.inf and cut != np.inf
    
    return feature, cut, bestloss

def cart(xTr,yTr,depth=np.inf,weights=None):
    n,d = xTr.shape
    if weights is None:
        w = np.ones(n) / float(n)
    else:
        w = weights
    
    # TODO:
    if (depth == 0 or np.all(yTr == yTr[0]) or np.all(xTr == xTr[0])):
        return TreeNode(None, None, None, None, None, np.sum(yTr * w)/np.sum(w))
    feature = 0
    cut = 0
    bestloss = 0
    feature, cut, bestloss = sqsplit(xTr, yTr, w)
    
    leftxTr = xTr[:, feature] <= cut
    rightxTr = xTr[:, feature] > cut
    leftyTr = yTr[leftxTr]
    rightyTr = yTr[rightxTr]
    weightsL = w[leftxTr]
    weightsR = w[rightxTr]
    leftTree = cart(xTr[leftxTr], leftyTr, depth - 1, weightsL)
    rightTree = cart(xTr[rightxTr], rightyTr, depth - 1, weightsR)
    pred = np.sum(yTr * w)/np.sum(w)
    return TreeNode(leftTree, rightTree, None, feature, cut, pred)

def evaltreehelper(root,xTe, idx=[]):
    assert root is not None
    n = xTe.shape[0]
    pred = np.zeros(n)
    
    # TODO:
    if (root.left == None and root.right == None):
        return root.prediction
    if (xTe[root.cutoff_id] <= root.cutoff_val):
        return evaltreehelper(root.left, xTe)
    return evaltreehelper(root.right, xTe)


def evaltree(root,xTe):
    # TODO:
    n = xTe.shape[0]
    pred = np.zeros(n)
    for i in range(0, xTe.shape[0]):
        pred[i] = evaltreehelper(root, xTe[i])
    binary_pred = (pred >= 0.40).astype(int)
    return binary_pred

def forest(xTr, yTr, m, maxdepth=np.inf):   
    n, d = xTr.shape
    trees = []
    
    # TODO:
    for i in range(0, m):
        indices = np.random.choice(n-1, n-1, replace = True)
        trees.append(cart(xTr[indices], yTr[indices], maxdepth))
    return trees

def evalforest(trees, X, alphas=None):
    m = len(trees)
    n,d = X.shape
    if alphas is None:
        alphas = np.ones(m) / len(trees)
            
    pred = np.zeros(n)
    
    # TODO:
    for i in range(0, m):
        pred += alphas[i] * evaltree(trees[i], X)
    
    binary_pred = (pred >= 0.40).astype(int)
    return binary_pred



def boosttree(x, y, maxiter=100, maxdepth=2):
    import numpy as np
    
    n, d = x.shape
    weights = np.ones(n) / n
    forest = []
    alphas = []

    # Convert labels from 0,1 to -1,1
    y_converted = np.where(y == 0, -1, 1)

    for i in range(maxiter):
        tree = cart(x, y_converted, maxdepth, weights)
        preds = evaltree(tree, x)
        # Calculate weighted error considering the converted labels
        incorrect = preds != y_converted
        error = np.sum(weights[incorrect])

        if error == 0:
            alphas.append(float('inf'))  # Or a large number to avoid infinite weights
            forest.append(tree)
            break

        if error < 0.5:
            alpha = 0.5 * np.log((1 - error) / error)
            # Update weights; preds are already -1, 1; y_converted are -1, 1
            weights *= np.exp(-alpha * y_converted * preds)
            weights /= weights.sum()
            forest.append(tree)
            alphas.append(alpha)
        else:
            # If the first classifier is not better than random, stop
            if not forest:  # Ensure at least one tree is returned
                forest.append(tree)
                alphas.append(0)  # Minimum possible influence
            break

    return forest, alphas



def main():
  my_data = np.genfromtxt('train.csv', delimiter=',')
  xTr = my_data[1:, :-1]
  yTr = my_data[1:, -1]
  xTrIon  = xTr.T
  yTrIon  = yTr.flatten()

  val_data = np.genfromtxt('validation.csv', delimiter=',')
  xVal = val_data[1:, :-1]
  yVal = val_data[1:, -1]

  test_data = np.genfromtxt('test.csv', delimiter=',')
  xTest = test_data[1:, 1:]

  root = cart(xTr, yTr)

  trees = forest(xTr, yTr, 50)
  tr_err = np.mean(np.sign(evalforest(trees,xTr)) != yTr)

  print(evalforest(trees, xTest))

if __name__ == "__main__":
    main()