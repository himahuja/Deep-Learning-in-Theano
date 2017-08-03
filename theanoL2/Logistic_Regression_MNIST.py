import numpy
import theano
import theano.tensor as T


from six.moves import cPickle as pickle
import gzip
import os
import sys
import timeit

class LogisticRegression(object):
    """
    Multi-class Logistic regression Class
    """

    def __init__(self, input, n_in, n_out):
        """
        Initialize the parameters of the logistic regression
        INPUT : theano.tensor.TensorType
                It is one mini-batch, symbolic variable that describes the
                input of the architecture.
        N_IN  : int
                the dimension of space of each input
        N_OUT : int
                the dimension of space of the number of labels
        """
        # initializing the weight matrix W (size : [n_in, n_out])
        self.W = theano.shared(
                value = numpy.zeros(
                        (n_in, n_out),
                        dtype = theano.config.floatX),
                name = 'W',
                borrow = True
        )


        # initializing the vector of biases b, as a vector of n_outs (value: [0])
        self.b = theano.shared(
                value = numpy.zeros((n_out,),
                        dtype = theano.config.floatX ),
                name = 'b',
                borrow = True
        )

        # SYMBOLIC EXPRESSION FOR COMPUTING THE MATRIX of
        # CLASS-MEMBERSHIP Probabilities
            # W is a matrix, where k-th column reprents the sepearation
            # hyperplane for class-k
            # x is a matrix, where j-th row represents input training sample - j
            # b is a vector where k-th element represents free parameter of
            # k-th hyperplane

        self.p_y_given_x = T.nnet.softmax (T.dot (input, self.W) + self.b)

        # symbolic description of how to compute prediction as
        # class whose probability is maximal.

        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

        # Storing the parameters of the model
        self.params = [self.W, self.b]

        # keep track of the model inputs :
        self.input = input


    def negative_log_likelihood(self, y):
        """
        Return the mean of the negative-log-likelihood of the prediction
        given target distribution P(y'|y)

        y : theano.tensor.TensorType
            corresponds to the vector that gives the correct label for each example
        """

        # Negative log-likelihood, for multi-class logistic regression
            # y.shape[0] -> number of rows in y i.e # of examples in mini-batch (n)
            # T.arange(y.shape[0]) vector containing [0, 1, 2, .... n-1]
            # Let T.log(self.p_y_given_x) be called Log Probabilities or (LP)
            # LP will have one row per example and one column per class
            # LP [T.arange(y.shape[0], y)] is a vector containing
                # [ LP[0, y[0]] , LP[1, y[1]], LP[2, y[2]], ... LP[n-1, y[n-1]] ]

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Returns the 0-1 loss over the size of the mini-batch
        y : theano.tensor.TensorType
            corresponds to the vector that gives the correct label for each example
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                            'y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean (T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def load_data(dataset):
    """
        loads the dataset
        dataset : string
                  represents the path of the dataset
    """
    ########################
    ####### LOAD DATA ######
    ########################

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # check if the dataset is in the data directory
        new_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
                dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(new_path)) or data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' %origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data!')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding = 'latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format : tuple (input, target)
    # input is a numpy.ndarray of 2 dimensions : (A Matrix)
    # where each row corresponds to an example.
    # Target is a nnumpy.ndarray of 1 dimension (vector) that has the
    # same number of rows as the `input matrix`

    def shared_dataset(data_xy, borrow = True):
        """
        Function that loads the dataset into the shared variables. Instead of
        copying every mini-batch to the GPU memory (which is slow and would
        significantly decrease performance)
        """
        data_x, data_y = data_xy
        shared_x = theano.shared (numpy.asarray(data_x,
                                                dtype = theano.config.floatX),
                                  borrow = borrow)
        shared_y = theano.shared (numpy.asarray(data_y,
                                                dtype = theano.config.floatX),
                                  borrow = borrow)

        # On GPU, the data is stored as floats (both the labels and the input data)
        # But during our computations we nee y_shared as ints. So we have to cast
        # them to integers.

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x,  test_set_y  = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

    return rval

def sgd_optimizations_mnist (learning_rate = 0.13, n_epochs = 1000,
                             dataset = 'mnist.pkl.gz',
                             batch_size = 600):
    """
    Demonstrates stochastic gradient descent optimization of a log-linear
    model.

    learning_rate : float, parameter supplied to the gradient equation.
    n_epochs      : int, number of times to run the optimizer on the whole data.
    dataset       : string, the path of the MNIST dataset file from,
                http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    # compute the number of mini-batches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow = True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]
    n_test_batches  = test_set_x.get_value(borrow = True).shape[0]

    #####################
    # BUILD ACTUAL MODEL #
    #####################

    print (' Building the model ...')

    # allocate symbolic variables for the data
    index = T.lscalar()

    # Generate symbolic varibales for input : x and labels : y
    x = T.matrix('x')
    y = T.ivector('y')

    # construct the LogisticRegression class
    # each MNIST image has size 28*28
    classifier = LogisticRegression(input = x, n_in = 28*28, n_out = 10)

    # we will minimize the negative log log-likelihood
    cost = classifier.negative_log_likelihood(y)

    # computing the gradient wrt to W and b
    g_W = T.grad(cost = cost, wrt = classifier.W)
    g_b = T.grad(cost = cost, wrt = classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x : train_set_x[ index * batch_size : (index + 1) * batch_size],
            y : train_set_y[ index * batch_size : (index + 1) * batch_size]
            }
        )

    test_model = theano.function(
        inputs = [index],
        outputs= classifier.errors(y),
        givens = {
            x : test_set_x[index * batch_size : (index + 1) * batch_size],
            y : test_set_y[index * batch_size : (index + 1) * batch_size]
            }
        )

    validate_model = theano.function(
        inputs = [index],
        outputs= classifier.errors(y),
        givens = {
            x : valid_set_x[index * batch_size : (index + 1) * batch_size],
            y : valid_set_y[index * batch_size : (index + 1) * batch_size]
            }
        )

    #################
    ## TRAIN MODEL ##
    #################

    print ('Training the model ...')
    # Early stopping parameters

    """
    Early Stopping Procedure
    We'll have patience about the improvement in performance,
    after the patience is over.

    Early stopping rules provide guidance as to how many iterations can be
    run before the learner begins to over-fit.
    """

    # look at these many examples before patience is up
    patience = 5000

    # wait this much longer when a new best is found
    patience_increase = 2

    improvement_threshold = 0.995
    # a relative improvement of this much is considered significant

    validation_frequency = min(n_train_batches, patience // 2)
    # go through these many mini-batches before checking the network
    # on the validation set ; in this case we check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.

    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while(epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            #iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                    for i in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *\
                                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score  = numpy.mean(test_losses)

                    print(
                        (
                            'epoch %i, minibatch %i / %i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_test_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model_mnist.pkl', 'wb') as f:
                        pickle.dump (classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print (
        (
            'Optimiation complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )

    print ('The code runs for %d epochs, with %f epochs/sec '%(
        epoch, 1. * epoch / (end_time - start_time) ))

    print (('The code for file ' + os.path.split(__file__)[1] +
            'ran for %.1fs') %((end_time - start_time)))


def predict():
    """
        Loading the trained model and use it to
        predict labels
    """
    # load the saved model
    classifier = pickle.load(open('best_model_mnist.pkl', 'rb'))

    # compile a predictor function
    predict_model = theano.function(
                        inputs = [classifier.input],
                        outputs=classifier.y_pred)

    # Testing on some examples from the test set
    dataset = 'mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    predicted_values = predict_model(test_set_x[:10])
    print('Predicted values for the first 10 examples in test set: ')
    print(predicted_values)


if __name__ == '__main__':
    sgd_optimizations_mnist()
