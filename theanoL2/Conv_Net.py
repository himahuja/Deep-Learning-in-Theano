# Imagine that layer m-1 is the input retina. In the above figure,
# units in layer m have receptive fields of width 3 in the input retina
# and are thus only connected to 3 adjacent neurons in the retina layer.
# Units in layer m+1 have a similar connectivity with the layer below.
# We say that their receptive field with respect to the layer below is also 3,
# but their receptive field with respect to the input is larger (5).
# Each unit is unresponsive to variations outside of its receptive field
# with respect to the retina. The architecture thus ensures that the
# learnt “filters” produce the strongest response to a spatially local
# input pattern

# However, as shown above, stacking many such layers leads to (non-linear)
# “filters” that become increasingly “global” (i.e. responsive to a larger region
# of pixel space). For example, the unit in hidden layer m+1 can encode a
# non-linear feature of width 5 (in terms of pixel space).

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy

import pylab
from PIL import Image

rng = numpy.random.RandomState(23455)

# input 4d Tensor
    # shape [mini-batch-size,
    #       number of input feature maps,
    #       image height,
    #       image width]

#instantiating a 4D Tensor for input
input = T.tensor4(name = 'input')

# INPUT HAS THE FOLLOWING PROPERTIES :
    # mini-batch-size = 2
    #
w_shp = (2, 3, 9, 9)
w_bound = numpy.sqrt(3 * 9 * 9)

# Constructing 4D Tensor corresponding to the weight matrix W
W = theano.shared(numpy.asarray(
                rng.uniform(
                    low = -1.0 / w_bound,
                    high = 1.0 / w_bound,
                    size = w_shp
                )
                dtype = input.dtype, name = 'W')

b_shp = (2,)
b= theano.shared( numpy.asarray(
                    rng.uniform(low=-.5, high=.5, size=b_shp),
                    dtype = input.dtype), name = 'b')


# Build the symbolic expression that computes the convolutions of input
# with filters in w
conv_out = conv2d(input, W)

# Build the symbolic expression to add bias and apply activation,
# i.e. it will produce the nnet layer output.


output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
f = theano.function([input], output)


##################################################
############# FUN WITH CONVOLUTIONS ##############
##################################################

img = Image.open(open('../data/wolf_growling.jpg'))

img = numpy.asarray(img, dtype = 'float64') / 256

    # Putting image in a 4D Tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_img = f(img_)

    # plot
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()
