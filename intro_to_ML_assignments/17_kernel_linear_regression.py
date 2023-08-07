#! .\intro_to_ML_venv\Scripts\python.exe

import argparse

import numpy as np
import sklearn.metrics

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
parser.add_argument("--data_size", default=50, type=int, help="Data size")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=5, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=50, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--iterations", default=200, type=int, help="Number of training iterations")
parser.add_argument("--l2", default=0.02, type=float, help="L2 regularization weight")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artifical regression dataset
    train_data = np.linspace(-1, 1, args.data_size)
    train_target = np.sin(5 * train_data) + generator.normal(scale=0.25, size=args.data_size) + 1

    test_data = np.linspace(-1.2, 1.2, 2 * args.data_size)
    test_target = np.sin(5 * test_data) + 1

    betas = np.zeros(args.data_size).reshape(-1,1)
    bias = np.mean(train_target)
    # TODO: Perform `args.iterations` of SGD-like updates, but in dual formulation
    # using `betas` as weights of individual training examples.
    #
    # We assume the primary formulation of our model is
    #   y = phi(x)^T w + bias
    # and the loss in the primary problem is batched MSE with L2 regularization:
    #   L = sum_{i \in B} 1/|B| * [1/2 * (phi(x_i)^T w + bias - target_i)^2] + 1/2 * args.l2 * w^2
    # Regarding the L2 re gularization, note that it always affects all betas, not
    # just the ones in the batch.
    #
    # For `bias`, explicitly use the average of the training targets, and do
    # not update it further during training.
    #
    # Instead of using feature map `phi` directly, we use a given kernel computing
    #   K(x, y) = phi(x)^T phi(y)
    # We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    #
    # After each iteration, compute RMSE both on training and testing data.

    def kernel_fce(x, z, degree, gamma, typ):
        x = x.reshape(-1,1)
        z = z.reshape(-1,1)
        if typ == "poly":
            kernel = (gamma * x @ z.T + 1)**degree
        elif typ == "rbf":
            kernel = np.exp(-gamma * np.sum((z-x[:, np.newaxis])**2, axis=-1))  
            #neco ze stackoverflow, to nasledujici funguje vyrazne pomaleji, ale je to prehledne
            
            #kernel = np.zeros((x.shape[0],z.shape[0]))
            #for i,xko in enumerate(x):
            #    for j,zko in enumerate(z):
            #        kernel[i,j] = np.exp(-gamma*np.linalg.norm(xko-zko)**2)
        return kernel
    
    train_rmses, test_rmses = [], []
    train_target = train_target.reshape(-1, 1)

    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`, performing
        # batched updates to the `betas`. You can assume that `args.batch_size`
        # exactly divides `train_data.shape[0]`.

        # TODO: Append RMSE on training and testing data to `train_rmses` and
        # `test_rmses` after the iteration.
        
        #vse shufflujeme, dulezite je ze v kernelu jsou shufflovany oba argumenty!
        train_data_shuffled = train_data[permutation]
        train_target_shuffled = train_target[permutation]
        betas_shuffled = betas[permutation]
        kernel_shuffled = kernel_fce(train_data_shuffled, train_data_shuffled, args.kernel_degree, args.kernel_gamma, args.kernel)
        
        loss = 0
        for batch in range(int(train_data.shape[0] / args.batch_size)):
            
            #v nasledujicim pocitam gradient pro bety daneho batche.
            gradient = np.empty((args.batch_size, 1))
            for dato in range(args.batch_size):
                
                gradient[dato] = ((betas_shuffled.T @ kernel_shuffled[batch * args.batch_size + dato, :].reshape(-1,1)) + bias
                                   - train_target_shuffled[batch * args.batch_size + dato]
                                 )
            
            #Tady pro ten jeden batch ty bety updatuju podle toho gradientu
            betas_shuffled[batch * args.batch_size : (batch + 1)*args.batch_size] -= args.learning_rate / args.batch_size * gradient
                                                #+ args.learning_rate*args.l2*betas_shuffled[batch*args.batch_size : (batch+1)*args.batch_size])
                
            #l2 regularizace! Trochu counter-intuitively se l2 regularizaci pri updatovani kazdeho batche zmensi vzdy VSECHNY bety
            betas_shuffled -= args.learning_rate * args.l2 * betas_shuffled
                
        # Z permutovanych bet udelame nepermutovane
        betas[permutation] = betas_shuffled
        
        # Vypocet RMSE train
        kernel = kernel_fce(train_data, train_data, args.kernel_degree, args.kernel_gamma, args.kernel)
        rmse_train = sklearn.metrics.mean_squared_error(train_target, kernel @ betas + bias, squared = False)
        #rmse_train = np.sqrt((rmse_train + (1/2*args.l2* betas.T @ betas)[0])[0]) # je nutne nastavit squared = True ...TAKHLE TO ALE STRAKA NECHTEL
        
        #Vypocet RMSE test
        test_kernel = kernel_fce(test_data, train_data, args.kernel_degree, args.kernel_gamma, args.kernel)
        rmse_test = sklearn.metrics.mean_squared_error(test_target, test_kernel @ betas + bias, squared = False)

        train_rmses.append(rmse_train)
        test_rmses.append(rmse_test)

        if (iteration + 1) % 10 == 0:
            print("Iteration {}, train RMSE {:.2f}, test RMSE {:.2f}".format(
                iteration + 1, train_rmses[-1], test_rmses[-1]))
        
    if args.plot:
        import matplotlib.pyplot as plt
        # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.
        test_predictions = test_kernel @ betas + bias

        plt.plot(train_data, train_target, "bo", label="Train target")
        plt.plot(test_data, test_target, "ro", label="Test target")
        plt.plot(test_data, test_predictions, "g-", label="Predictions")
        plt.legend()
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return train_rmses, test_rmses

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)