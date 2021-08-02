import numpy as np
from loss import *
from layer_utils import *

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        prev_hidden_dim = None
        
        for i in range(self.num_layers):
          
          if i== self.num_layers-1:
            w = np.random.randn(prev_hidden_dim,num_classes) * weight_scale
            b = np.zeros((num_classes,))
          else:
            hidden_dim = hidden_dims[i]
            b = np.zeros((hidden_dim,))

            if i == 0:
              w = np.random.randn(input_dim, hidden_dim) * weight_scale
            else:
              w = np.random.randn(prev_hidden_dim, hidden_dim) * weight_scale
          
          if self.use_batchnorm and i < self.num_layers-1:
            gamma = np.ones_like(b) 
            beta = np.zeros_like(b)
            gamma_params_name = 'gamma'+str(i+1)
            beta_params_name = 'beta'+str(i+1)

            self.params[gamma_params_name] = gamma
            self.params[beta_params_name] = beta

          prev_hidden_dim = hidden_dim
          w_params_name = 'w'+str(i+1)
          b_params_name = 'b'+str(i+1)
          
          self.params[w_params_name] = w
          self.params[b_params_name] = b
         

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        cache_list = []
        relu_cache_list = []
        bn_cache_list = []

        out = None
        for i in range(self.num_layers):
            w = self.params['w'+str(i+1)]
            b = self.params['b'+str(i+1)]
            
           
            if i == self.num_layers -1:
              scores,cache = affine_forward(out,w,b)
            
            else:
              if i ==0:
                out,cache = affine_forward(X,w,b)
              else:
                out,cache = affine_forward(out,w,b)

              if self.use_batchnorm:
                out,bn_cache = batchnorm_forward(out,self.params['gamma'+str(i+1)],self.params['beta'+str(i+1)],self.bn_params[i])
                bn_cache_list.append(bn_cache)
                
              out,rn_cahce = relu_forward(out)
              relu_cache_list.append(rn_cahce)
            
            cache_list.append(cache)      
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test' or y is None:
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss(scores,y)
        dx = None
        for i in range(self.num_layers,0 ,-1):
            w_name = 'w'+str(i)
            b_name = 'b'+str(i)

            w = self.params[w_name]
           
          
            if i ==self.num_layers:
              dx,dw,db = affine_backward(dout,cache_list[i-1])
            else:
              dx = relu_backward(dx,relu_cache_list[i-1])
              
              if self.use_batchnorm:
                dx,dgamma,dbeta = batchnorm_backward(dx,bn_cache_list[i-1])
                gamma_name = 'gamma'+str(i)
                beta_name = 'beta'+str(i)
                
                grads[gamma_name] = dgamma
                grads[beta_name] = dbeta 

              dx,dw,db = affine_backward(dx,cache_list[i-1])
              
             

              

            grads[w_name] = dw + self.reg * w 
            grads[b_name] = db
          
            loss += self.reg*np.sum(w*w) * 0.5
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

