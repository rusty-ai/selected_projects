#!/usr/bin/env python3

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=3, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions") 
################-----------------------------------------------------------
#BACHA! Nejak mi to zkolabuje pro rbf kernel kdyz dam plotovat, pry "operands could not be broadcast together with shapes (51,2) (2,1)"
#Tzn mam nejak spatne zadefinovany rbf kernel, ktery nebude fungovat pro liche x? I kdyz jsem zkousel variantu:
            #kernel = np.zeros((x.shape[0],z.shape[0]))
            #for i,xko in enumerate(x):
            #    for j,zko in enumerate(z):
            #        kernel[i,j] = np.exp(-gamma*np.linalg.norm(xko-zko)**2)
# ktera je neefektivni ale blbu vzdorna, tak mi to nefungovalo. Cim to muze byt? Jak to vyresit jinak? Uloha jinak funguje dobre.
################-----------------------------------------------------------
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.


def kernel(args: argparse.Namespace, x: np.ndarray, y: np.ndarray): # -> np.ndarray:
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    
    # Zadefinovane radeji v samotnem smo()

    raise NotImplementedError()

# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
): # -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    def kernel_fce(x, z):
        degree, gamma, typ = args.kernel_degree, args.kernel_gamma, args.kernel
        #x = x.reshape(-1,1)  TOHLE TU NEMUZE BYT, protoze train/test_data maji dvoje featury (meli jsme to v kernel_linear_regression)
        #z = z.reshape(-1,1)  
        if typ == "poly":
            kernel = (gamma * x @ z.T + 1)**degree
        elif typ == "rbf":
            kernel = np.exp(-gamma*np.sum((z-x[:, np.newaxis])**2, axis=-1))
        return kernel
    
    kernel = kernel_fce(train_data, train_data)
    
    passes_without_as_changing = 0
    train_accs, test_accs = [], []
    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)
            
            E_i = (kernel[i] @ np.multiply(a,train_target) + b) - train_target[i] # y(xi) - ti. "a" jsou neco jako betas, "b" je neco jako bias
            
            if ( a[i] < args.C - args.tolerance and train_target[i]*E_i < -args.tolerance ) or ( a[i] > args.tolerance and train_target[i]*E_i > args.tolerance ):
                E_j = (kernel[j] @ np.multiply(a,train_target) + b) - train_target[j]
                
                #druha derivace lagranzianu
                druha_d_L = 2 * kernel[i, j] - kernel[i, i] - kernel[j, j]
                if druha_d_L < args.tolerance:    # hledame maximum, tzn 2. der. musi byt negativni (viz napr. kvadr. fce)
                    a_j_new = a[j] - train_target[j]*(E_i - E_j) / druha_d_L
                    
                    #Ted musim zjistit jestli to a_j_new je v te ohradce nebo ne, a "orezat" to prip podle okraju, navic zapocitat a_i_new. Viz slide 19, 7. prednaska
                    if train_target[i] == train_target[j]:
                        #omezim to novym intervalem [L,H], kde v tech okrajovych bodech L a H je zapocitany vliv toho a_i_new
                        L = np.maximum(0, a[i] + a[j] - args.C)
                        H = np.minimum(args.C, a[i] + a[j])
                        if a_j_new < L:
                            a_j_new = L
                        if a_j_new > H:
                            a_j_new = H
                        
                    elif train_target[i] != train_target[j]:
                        #druha varianta toho predchoziho pro pripad ze se targety porovnavanych hodnot lisi
                        L = np.maximum(0, a[j] - a[i])
                        H = np.minimum(args.C, args.C + a[j] - a[i])
                        if a_j_new < L:
                            a_j_new = L
                        if a_j_new > H:
                            a_j_new = H
                    
                    #Nyni jdu zjistit jestli se mi hodnota pro a_j_new dostatecne zmenila. Pokud ne, zahodim vysledky a pokracuji k dalsimu i
                    if not np.abs(a_j_new - a[j]) < args.tolerance :
                    
                        #Nyni dopocitat novou hodnotu pro a[i], pote bias b - ten spoctu tak ze urcim dva biasy b1 a b2 a z nich teprve to b
                        a_i_new = a[i] - train_target[i]*train_target[j]*(a_j_new - a[j])
                    
                        b_j_new = b - E_j - train_target[i]*(a_i_new - a[i])*kernel[i,j] - train_target[j]*(a_j_new - a[j])*kernel[j,j]
                        b_i_new = b - E_i - train_target[i]*(a_i_new - a[i])*kernel[i,i] - train_target[j]*(a_j_new - a[j])*kernel[j,i]
                    
                        if args.tolerance < a_i_new and a_i_new < args.C - args.tolerance:
                            b_new = b_i_new
                        elif args.tolerance < a_j_new and a_j_new < args.C - args.tolerance:
                            b_new = b_j_new
                        else: 
                            b_new = (b_i_new + b_j_new)/2
                        
                        #Nyni updatuju a[i], a[j] i b a zvetsim "as_changed"
                        a[j] = a_j_new
                        a[i] = a_i_new
                        b = b_new
                        as_changed += 1

            # TODO: Check that a[i] fulfils the KKT conditions, using `args.tolerance` during comparisons.

            # If the conditions do not hold, then
            # - compute the updated unclipped a_j^new.
            #
            #   If the second derivative of the loss with respect to a[j]
            #   is > -`args.tolerance`, do not update a[j] and continue
            #   with next i.

            # - clip the a_j^new to suitable [L, H].
            #
            #   If the clipped updated a_j^new differs from the original a[j]
            #   by less than `args.tolerance`, do not update a[j] and continue
            #   with next i.

            # - update a[j] to a_j^new, and compute the updated a[i] and b.
            #
            #   During the update of b, compare the a[i] and a[j] to zero by
            #   `> args.tolerance` and to C using `< args.C - args.tolerance`.

            # - increase `as_changed`

            
        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.

        train_target_predict = np.sign(kernel @ np.multiply(a,train_target) + b)
        
        test_kernel = kernel_fce(test_data, train_data)
        test_target_predict = np.sign(test_kernel @ np.multiply(a,train_target) + b)

        train_accs.append(sklearn.metrics.accuracy_score(train_target_predict, train_target))
        test_accs.append(sklearn.metrics.accuracy_score(test_target_predict, test_target))
        
        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
                         
    #support_vectors = np.select(a > args.tolerance, a)

    #condition = a > args.tolerance
    #support_vectors = np.extract(condition, a)

    support_vectors, support_vector_weights = [], []
    for i in range(a.shape[0]):
        if a[i] > args.tolerance:
            support_vectors.append(train_data[i])
            support_vector_weights.append(a[i]*train_target[i])
    support_vectors = np.array(support_vectors)
    support_vector_weights = np.array(support_vector_weights)

    print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

    return support_vectors, support_vector_weights, b, train_accs, test_accs

def main(args: argparse.Namespace): # -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt
        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM value `y(x)` for the given x.

        def kernel_fce(x, z):
            degree, gamma, typ = args.kernel_degree, args.kernel_gamma, args.kernel
            if typ == "poly":
                kernel = (gamma * x @ z.T + 1)**degree
            elif typ == "rbf":
                kernel = np.exp(-gamma*np.sum((z-x[:,np.newaxis])**2,axis=-1))
            return kernel
        
        predict_function = lambda x: np.sign(kernel_fce(x, support_vectors) @ support_vector_weights + bias)

        plot(predict_function, support_vectors)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)