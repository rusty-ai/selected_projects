#!/usr/bin/env python3

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=3, type=int, help="Number of SGD training epochs")
parser.add_argument("--hidden_layer", default=200, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace): # -> tuple[tuple[np.ndarray, ...], list[float]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]
    
    # TODO: Implement forward propagation, returning *both* the value of the hidden
    # layer and the value of the output layer.
    #
    # We assume a neural network with a single hidden layer of size `args.hidden_layer`
    # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
    # activation.
    #
    # The value of the hidden layer is computed as ReLU(inputs @ weights[0] + biases[0]).
    # The value of the output layer is computed as softmax(hidden_layer @ weights[1] + biases[1]).
    #
    # Note that you need to be careful when computing softmax, because the exponentiation
    # in softmax can easily overflow. To avoid it, you should use the fact that
    # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
    # That way we only exponentiate values which are non-positive, and overflow does not occur.
        
    #raise NotImplementedError()

    y_test=np.zeros((test_target.shape[0], 10))
    for dato in range(test_target.shape[0]):
        y_test[dato,test_target[dato]] = 1
        
    y_train=np.zeros((train_target.shape[0], 10))
    for dato in range(train_target.shape[0]):
        y_train[dato, train_target[dato]] = 1
    
    def ReLU(x):
        x[x < 0] = 0 #pripadne x=np.maximum(0,x)
        return x
    def der_ReLU(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    def softmax(x):
        #Compute softmax values for each sets of scores in x.
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis = 0)
    
    def forward(X,weights, biases, batch_size, pocatecni_dato):       #forward(inputs):
        #X je CELY ten X_train ci X_test dataset, pocatecni dato mi rika od ktereho data v datasetu chci videt vystupy hidden a output vrstvy
        #a batch size mi da interval po ktery chci ty vystupy videt. Vystupem jsou dve matice o shape (m,n), kde n jsou hodnoty neuronu tech vrstev
        #pro dane vzorky (resp. data) z toho datasetu, kterych je n=batch_size. 
        hidden_layers=np.empty((batch_size, weights[0].shape[1]))
        output_layers=np.empty((batch_size, weights[1].shape[1]))
        for dato in range(batch_size):
            hidden_layer=ReLU(X[pocatecni_dato + dato] @ weights[0] + biases[0])
            output_layer=softmax(hidden_layer @ weights[1] + biases[1])
            hidden_layers[dato] = hidden_layer
            output_layers[dato] = output_layer
        return hidden_layers, output_layers
        
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        perm_X_train = train_data[permutation]
        perm_y_train = y_train[permutation]

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        
        for batch in range(int(train_data.shape[0] / args.batch_size)):
            hidden_layers,output_layers=forward(perm_X_train, weights, biases, args.batch_size, batch * args.batch_size)

        # The gradient used in SGD has now four parts, gradient of weights[0] and weights[1]
        # and gradient of biases[0] and biases[1].
        
        #vstup X --> pošlu na něj váhy w_h a bias b_h --> mám h_in --> aplikuji ReLU --> mám h
        # --> váhy w_y a bias b_y --> mám y_in --> softmax --> mám výstup y. K němu náleží loss L.
        #Potrebuju spocitat gradienty lossu podle vah a biasu, abych je mohl updatovat.
        #Viz video cviceni 5 od Straky.
        
            grad_by = 0
            grad_wy = 0
            grad_bh = 0
            grad_wh = 0
            for dato in range(args.batch_size):
                
                #!!! Samozrejme by to nasledujici slo zkratit, takto to ale presne odpovida tomu postupu od straky, je to nazorne
                
                #Nejprve gradienty pro weights[1] a biases[1]
                dL_dyin = output_layers[dato] - perm_y_train[batch * args.batch_size+dato]
                dL_dby = dL_dyin #protoze dL/dby=dL/dyin*dyin/dby kde ten druhy clen je rovny jedne
                dL_dwy = hidden_layers[dato].reshape(-1, 1) @ dL_dyin.reshape(-1, 1).T
                
                grad_by += dL_dby / args.batch_size
                grad_wy += dL_dwy / args.batch_size
                
                #Dale derivace Loss podle vystupu z hidden layer, pak derivace vystupu z hidden layer podle vstupu do hidden layer, 
                #nebot potrebuji derivaci loss podle vstupu do hidden layer
                dL_dh = weights[1] @ dL_dyin   #np.dot(weights[1],dL_dyin)
                dh_dhin = der_ReLU(hidden_layers[dato])
                #dL_dhin=np.matmul(dL_dh,dh_dhin)
                dL_dhin = dL_dh.reshape(-1, 1) * dh_dhin.reshape(-1, 1)   #akorat pak je treba k dL_dbh treba pridat 
                                                                    #.reshape(args.hidden_layer) shape (x,1) na (x,)
                
                #jelikoz mam uz derivaci loss podle vstupu do hidden layer, mohu spocitat gradient weights[0] a biases[0] analogicky k predchozim vaham.
                dL_dbh = dL_dhin.reshape(args.hidden_layer)
                dL_dwh = perm_X_train[batch * args.batch_size + dato].reshape(-1, 1) @ dL_dhin.reshape(-1, 1).T
                
                grad_bh += dL_dbh / args.batch_size
                grad_wh += dL_dwh / args.batch_size

            weights[1] = weights[1] - args.learning_rate * grad_wy
            biases[1] = biases[1] - args.learning_rate * grad_by
            
            weights[0] = weights[0] - args.learning_rate * grad_wh
            biases[0] = biases[0] - args.learning_rate * grad_bh
            
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of -log P(target | data), or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer
        # - compute the derivative with respect to weights[1] and biases[1]
        # - compute the derivative with respect to the hidden layer output
        # - compute the derivative with respect to the hidden layer input
        # - compute the derivative with respect to weights[0] and biases[0]
        
        #Toto potrebuju na to abych z vystupu y=(0.1,0,0.8,0.1,...,0) udelal (0,0,1,0,...,0)
        def pomocna_fce_accuracy(y):
            y_nove = np.zeros(y.shape)
            for dato in range(y.shape[0]):
                maximum = np.argmax(y[dato])
                y_nove[dato, maximum] = 1
            return y_nove
        
        
        #Nejprve spoctu test_accuracy
        hidden_layers, pred_y_test = forward(test_data, weights, biases, test_data.shape[0], 0)
        
        test_accuracy = sklearn.metrics.accuracy_score(y_test, pomocna_fce_accuracy(pred_y_test))   #np.round(pred_y_test)) nelze pouzit,
                                                                                                    #z (0.4,0.45,0.15) to udela (0,0,0), chci (0,1,0)
        """
        suma=0
        for dato in range(y_test.shape[0]):
            suma+=np.dot(y_test[dato],pred_y_test[dato])  #<---- toto by fungovalo, jen pred_y_test by muselo byt prohnano obdobou pomocne_fce_accuracy
        test_accuracy=suma/y_test.shape[0]
        """
        #Ted spoctu train_accuracy
        hidden_layers, pred_y_train = forward(train_data, weights, biases, train_data.shape[0], 0)
        
        train_accuracy = sklearn.metrics.accuracy_score(y_train, pomocna_fce_accuracy(pred_y_train))
        
        # TODO: After the SGD epoch, measure the accuracy for both the
        # train test and the test set.

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))
    
    return tuple(weights + biases), [train_accuracy, test_accuracy]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(args)
    print("Learned parameters:", *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:20]] + ["..."]) for ws in parameters), sep="\n")
    #Learned parameters jsou na radcich poporade: weights[0], weights[1], biases[0], biases[1]
