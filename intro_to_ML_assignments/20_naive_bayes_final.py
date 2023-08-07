#!/usr/bin/env python3

import argparse
import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics
import scipy.stats

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=10, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

#----------- DVOJE DEFINICE
def variance(vsechna_data_jedne_tridy):
    mean = np.mean(vsechna_data_jedne_tridy, axis = 0)
    output_pred_korekci = 1/vsechna_data_jedne_tridy.shape[0]*np.sum((vsechna_data_jedne_tridy - mean)**2, axis = 0)
    return  output_pred_korekci #resp tedy tohle je kvadrat variance myslim (edit behem zkouskoveho)

def GaussianProb(train_mean, train_variance, test_data):
    exponent = np.exp(-((test_data-train_mean)**2 / (2 * train_variance**2 )))
    return (1 / (np.sqrt(2 * np.pi) * train_variance)) * exponent
#-----------
    
def main(args: argparse.Namespace) -> float:
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    #---------------------------------------------------------------MUJ KOD
    
    ###Nejprve spocteme prior
    prior = np.zeros(np.unique(train_target).shape)
    for j, trida in enumerate(np.unique(train_target)):
        prior[j] = len(train_target[train_target == trida])/len(train_target)

    if args.naive_bayes_type == "gaussian":

    ###Rozdelime train data podle trid a pro kazdou tridu najdeme mean a varianci kazde featury ("fit" cast modelu)
        train_mean = np.empty( (len(np.unique(train_target)), train_data.shape[1]) )    ###shape train_mean a train_var je (# trid, # featur)
        train_variance = np.empty( (len(np.unique(train_target)), train_data.shape[1]) )
        for index, trida in enumerate(np.unique(train_target)):
            train_data_dane_tridy = train_data[train_target == trida]
            train_mean[index] = np.mean(train_data_dane_tridy, axis = 0)
            train_variance[index] = variance(train_data_dane_tridy)
            #train_variance[index] = np.var(train_data_dane_tridy, axis = 0)   ###.. taky funguje ale Straka to chtel z definice.

    ### "predict" cast modelu. 
        predikce = []
        for i, dato in enumerate(test_data):
            #conditional = np.sum(scipy.stats.norm.logpdf(test_data[i], loc=train_mean, scale=np.sqrt(train_variance + args.alpha)), axis = 1)
            conditional = np.sum(np.log(GaussianProb(train_mean, np.sqrt(train_variance + args.alpha), test_data[i])), axis = 1)
            predikce.append( np.argmax(np.log(prior) + conditional))
    
    
    if args.naive_bayes_type == "bernoulli":
        
        ###Nejprve potrebujeme binarizovat featury
        binarized_train_data = np.where(train_data >= 8, 1, 0)
        
        ### "fit" cast modelu - musime urcit pravdepodobnosti p_d_k (prezentace 8, slide 26 --> https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/slides/?08#95)
        matice_pravdepodobnosti = np.zeros((len(np.unique(train_target)), train_data.shape[1]))
        for index, trida in enumerate(np.unique(train_target)):
            binarized_train_data_dane_tridy = binarized_train_data[train_target == trida]
            neupravena_matice_pravdepodobnosti = np.mean(binarized_train_data_dane_tridy, axis = 0)
            matice_pravdepodobnosti[index] = (neupravena_matice_pravdepodobnosti * binarized_train_data_dane_tridy.shape[0] + 
                                              args.alpha) / (binarized_train_data_dane_tridy.shape[0] + 2*args.alpha)
        
        ### "predict" cast modelu - predikce je soucet log(prioru) + conditional, coz log te distribuce
        ### (viz prezentace 8, slide 24, treti vzorec --> https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/slides/?08#88)
        predikce = []
        binarized_test_data = np.where(test_data >= 8, 1, 0)
        for i, binarized_test_dato in enumerate(binarized_test_data):
            conditional = np.sum(binarized_test_dato * np.log(matice_pravdepodobnosti/(1-matice_pravdepodobnosti)) + 
                                      np.log(1 - matice_pravdepodobnosti), axis = 1)
            predikce.append(np.argmax(np.log(prior) + conditional))
            
            
    if args.naive_bayes_type == "multinomial":
        
        ### "predict" cast modelu - opet hledame matici pravdepodobnosti p_d_k - viz prezentace 8, slide 29 --> https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/slides/?08#103
        matice_pravdepodobnosti = np.zeros((len(np.unique(train_target)), train_data.shape[1]))
        for index, trida in enumerate(np.unique(train_target)):
            train_data_dane_tridy = train_data[train_target == trida]
            matice_pravdepodobnosti[index] = (np.sum(train_data_dane_tridy, axis = 0) + args.alpha) / (np.sum(train_data_dane_tridy) + args.alpha*train_data.shape[1])
        
        ### "fit" cast modelu - slide 27, prezentace 8 --> https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/slides/?08#97
        predikce = []
        for i, dato in enumerate(test_data):
            conditional = np.sum(dato * np.log(matice_pravdepodobnosti), axis = 1)
            predikce.append(np.argmax(np.log(prior) + conditional))

#-------------- PRO ZAJIMAVOST:
    """
To co mi počítáme je P(y|X) = P(X|y) * P(y) / P(X), kde P(X) je konst., tak to neuvažujeme
- P(y|X) je posterior = ta moje predikce. 
- P(y) je prior, což jen zohledňuje zastupení dat různých tříd v trénovacím datasetu. Tzn prior je vektor o shape (# tříd), kde prior[i] = # train_dat dané třídy / # train_dat
- P(X|y) (= conditional) je komplikovanější. Z trénovacích dat si udělám pro každou třídu a featuru distribuci - ta distribuce mi modeluje pravděpodobnost toho, že ta featura má 
pro danou třídu nějakou určitou hodnotu. Já pak pro určitou featuru nějakého testovacího data zjistím, jaká ta pravděpodobnost že ta featura má zrovna tu určitou hodnotu je. 
Tyto všechny pravděpodobnosti pro všechny featury pak všechny vynásobím a to je ten můj conditional P(X|y)
--> Protože ale to P(X|y) může být strašně malé, vyplatí se použít logaritmickou verzi toho P(y|X) = P(X|y) * P(y), tedy ln(P(y|X)) = ln(P(X|y)) + ln(P(y)). 
S tím pracuji u všech variant naive_bayes. 

U toho gaussian predict jsem si ale udělal predikci i pro tu nelog. metodu. Odkomentuj jen jednu matice_gaussianu - jedna pracuje s mojí ze stackoverflow upravenou
rovnicí pro gaussian distribuci, ta druhá využívá scipy knihovnu jak doporučoval Straka - obě řešení jsou ekvivalentní. 

            #matice_gaussianu = scipy.stats.norm.pdf(test_data[i], loc=train_mean, scale=np.sqrt(train_variance + args.alpha))
            #matice_gaussianu = GaussianProb(train_mean, np.sqrt(train_variance + args.alpha), test_data[i])
            #conditional = np.prod(matice_gaussianu, axis = 1)
            #predikce.append(np.argmax(prior * conditional))
    """         
    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute probability density function of a Gaussian
    #   distribution using `scipy.stats.norm`, which offers `pdf` and `logpdf`
    #   methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.

    # TODO: Predict the test data classes and compute test accuracy.

    test_accuracy = sklearn.metrics.accuracy_score(predikce, test_target)
    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))