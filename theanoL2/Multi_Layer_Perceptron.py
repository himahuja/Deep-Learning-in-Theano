import theano
import theano.tensor as T
import numpy
from Logistic_Regression import LogisticRegression, load_data
import os
import sys
import timeit
from six.moves import cPickle as pickle
# This program will focus on a single-hidden-layer MLP.
# We start off by implementing a class that will represent a hidden layer.
# To construct the MLP we will then only need to throw a
# logistic regression layer on top.

# NOTE: For tanh activation function the interval from which weights should be
#       randomly generated, between [- --> + sqrt(6/(fan_in + fan_out))]
#       For sigmoid : between [- --> + 4 * sqrt(6/(fan_in + fan_out))]

class HiddenLayer(object):
    def __init__ (self, rng, input, n_in, n_out, W = None, b=None, activation = T.tanh):
        """
        Typical hidden layer of an MLP: units are fully connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in, n_out)
        and the bias vector b is of shape (n_out,)

        NOTE: The nonlinearity used here is tanh
        Hidden unit activation is given by : tanh(dot(input, W) + b)

        rng     : numpy.random.RandomState
                  a random number generator used to initialize weights
        input   : theano.tensor.dmatrix
                  symbolic tensor of shape (n_examples, n_in)
        n_in    : int
                  dimesionality of input

        n_out   : int
                  number of hidden units

        activation: theano operation or function
                    Non linearity to be applied in the layer

        """

        self.input = input

        if W is None :
            W_values = numpy.asarray(
                rng.uniform(
                    low = -numpy.sqrt(6. / (n_in + n_out)),
                    high = numpy.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype = theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared (value = W_values, name='W', borrow = True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]


class MLP(object):
    """
    MLP is an FFNN, has one or more hidden layers
    and a nonlinear activation function.

    The final layer, topmost layer is a softmax function.
    """

    def __init__ (self, rng, input, n_in, n_hidden, n_out):
        """
        initialize the parameters of the perceptron.

        rng : numpy.random.RandomState
              random number generator to initialize weights
        input : theano.tensor.TensorType
                input to the architecture (one minibatch)

        n_in : int
               the number of features of the input variable
        n_hidden : int
                number of hidden units
        n_out : int
                number of output units
        """

        self.hiddenLayer = HiddenLayer(
            rng = rng,
            input = input,
            n_in = n_in,
            n_out = n_hidden,
            activation = T.tanh
        )


        self.logRegressionLayer = LogisticRegression(
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_out
        )
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input

def test_mlp (learning_rate = 0.01, L1_reg = 0.00, L2_reg = 0.0001, n_epochs = 1000,
              dataset = 'mnist.pkl.gz', batch_size = 20, n_hidden = 500):
    """
    Gradient descent on a multi-layer-perceptron
    learning_rate : float
                    factor for gradient descent
    L1_reg        : float, L1-Norm of weights
    L2_reg        : float, L2-Norm of weights
    n_epochs      : int, maximal number of epochs to run on the system
    dataset       : string, path to the MNIST dataset
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x , test_set_y  = datasets[2]

    # compute the number of mini-batches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow = True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow = True).shape[0] // batch_size
    n_test_batches  = test_set_x.get_value(borrow = True).shape[0] // batch_size

    #####################
    # BUILD ACTUAL MODEL #
    #####################

    print ('Building the model ...')

    # allocate symbolic variables for the data
    index = T.lscalar()

    # Generate symbolic varibales for input : x and labels : y
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)

    # Construct the MLP Class

    classifier = MLP(
        rng = rng,
        input = x,
        n_in = 28*28,
        n_hidden = n_hidden,
        n_out = 10
    )

    cost = (
            classifier.negative_log_likelihood(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr
    )

    # computing the gradient of cost with respect to theta
    gparams = [T.grad(cost, param) for param in classifier.params]

     # specifying the update expression as a list of tuples:
     # (variable, update expression) pairs

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)]

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
    while (epoch < n_epochs) and (not done_looping):
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
                    if this_validation_loss < best_validation_loss * improvement_threshold:
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
        epoch, 1. * epoch / (end_time - start_time) / 60.0 ))

    print (('The code for file ' + os.path.split(__file__)[1] +
            ' ran for %.2fm' % ((end_time - start_time))), file=sys.stderr)

if __name__ == '__main__':
    test_mlp()
