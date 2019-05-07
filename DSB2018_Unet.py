"""
@author: Christopher Bumgardner
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import copy

"""A u-net architecture for finding image masks. 

This is a standard u-net architecture for finding object masks. It outputs and 
trains on greyscale image data with objects' boundaries, interior, and exterior 
labeled with a one-hot label. The loss function for training has several  
parameters for customizing the loss. See the u_net_fn for a description of the 
loss parameters. Briefly, it's possible to attach a loss bias toward particular 
types of mislabeling (e.g. interior was labeled as exterior); attribute more 
loss to regions touching the boundaries; and weight the loss on boundary, 
interior and exterior differently. The bias toward particular mislabelings is 
for mislabeling that on the end of producing the masks in post processing cause 
more problems than other. For example, mislabeling a boundary point as an 
interior point might be worse than labeling it as an exterior point since two 
objects touching need to know that they have something distingushing their 
interiors, whether boundary or exterior. 

Postprocessing to make 2D masks and IOU evaluation for the model should be done 
with the use of the datasciencebowl2018 module functions. For example:
    from datasciencebowl2018 import boundary_knn_batch,fill_int_holes
    
    int_pred_masks = boundary_knn_batch(predicted_masks)
    int_true_masks = boundary_knn_batch(test_masks)
    fill = partial(fill_int_holes, area = 25, smoothing = 1, remove_objects = 4)
    int_pred_masks = [fill(int_pred_masks[i,...])[np.newaxis,...] for i in range(length)]
    int_pred_masks = np.concatenate(int_pred_masks)
    mean = [iou_metric(int_true_masks[i,...],int_pred_masks[i,...]) for i in range(length)]
    mean = np.mean(mean)
    print("Mean for test batch:{}".format(mean))
"""

def iou_metric(y_true_in, y_pred_in,print_table=False):
    """IOU metric function.
    
    The image data should be integer labeled regions for masked objects. 
    Background is labeled with 0, and other objects should be labeled with 
    integers > 0. The output is a mean over iou thresholds range(0.5,1,0.5).
    
    Args: 
        y_true_in: ndarray of integer labelings of true labelings
        y_pred_in: ndarray of integer labelings of predicted labelings
        print_table: boolean to produce a precision threshold table
    Returns: 
        IOU average value over thresholds.
    """
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(),
                                  bins=(true_objects, pred_objects))[0]

    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection

    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    iou = intersection / union

    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp = np.sum(true_positives)
        fp = np.sum(false_positives)
        fn = np.sum(false_negatives)
        return tp, fp, fn

    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec) 


class hyperparams(object):
    """The class object holding all params for the u-net model and training.
    
    Attributes: 
        layers: list of 5 integers for number of cell for each convolution layer
        bdy_cost: float overall cost for boundary labeling mistakes
        int_cost: float overall cost for interior labeling mistakes
        bgd_cost: float overall cost for background labeling mistakes
        fat_bdy_on: boolean to include extended boundary values into cost for 
            boundary
        use_bias: 3x3 bias matrix  ext int bdy <--incorrect categorization bias
                                    (1  b   c) ext true
                                    (d  1   f) int true
                                    (g  h   1) bdy true
                  values must be positive. see u_net_fn for more description.
        batch_size: trainging batch size
        training_steps: number of steps for training 
        learning_rate: float learning rate for ADAM,
        epsilon: standard epsilon for ADAM
        clip_norm: float for norm clipping
        regularizer: tf regularizer object for network cells
        activation: activation function for network cells
        rates: list of dropout probability rates for each convolution and conv 
            transpose layer. There should 9 values in list. 
        checkpoint_path: str for absolute filepath for checkpoints
        predict_batch_size: batch size for predicting mode
        save_checkpoints_steps: number of training steps before checkpoint save
        keep_checkpoint_max: number of checkpoints to save (first in first out)
        save_summary_steps: number of steps before saving model summary for 
            tensorboard
        logging_step: number of steps before next output log 
    
    """
    
    
    def __init__(self,
                 layers = [64, 128, 256, 512, 1028],
                 bdy_cost = 1,
                 int_cost = 1,
                 bgd_cost = 1,
                 fat_bdy_on = False,
                 use_bias = np.array([[1,1,1],[1,1,1],[1,1,1]]),
                 batch_size = 1,
                 training_steps = 1,
                 learning_rate = 0.001,
                 epsilon = 0.1,
                 clip_norm = 10**5,
                 regularizer = tf.contrib.layers.l2_regularizer(scale=0.1),
                 activation = tf.nn.elu,
                 rates = [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 checkpoint_path = None,
                 predict_batch_size = 30,
                 save_checkpoints_steps = 50,
                 keep_checkpoint_max = 20,
                 save_summary_steps = 10,
                 logging_step = 10,
                 ):
        """Initialization. 
        
        Raises: 
            ValueError: if the layers list is the incorrect size
            ValueError: if the list of dropout rates is incorrectly specified
            ValueError: if use_bias matrix has incorrect specifications
        """ 
        if len(layers) != 5:
                raise ValueError("Layers list must have 5 values.")
        else:
                self.layers = layers
    
        self.bdy_cost = bdy_cost
        self.int_cost = int_cost
        self.bgd_cost = bgd_cost
        self.fat_bdy_on = fat_bdy_on
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.regularizer = regularizer
        
        if len(rates) == 9 and np.all(np.array(rates) >= 0) and np.all(np.array(rates) <= 1):
                self.rates = rates
        else:
            raise ValueError("Rates must be a list of 9 values between 0 and 1.")
            
        self.activation = activation
        self.checkpoint_path = checkpoint_path
        self.save_checkpoints_steps = save_checkpoints_steps
        self.keep_checkpoint_max = keep_checkpoint_max
        self.save_summary_steps = save_summary_steps
        self.logging_step = logging_step 
        
        try:
            if use_bias.shape == (3,3):
                if np.all(use_bias > 0):
                    if not np.all(use_bias.diagonal() == [1,1,1]):
                        self.use_bias = use_bias - np.diag(np.diag(use_bias)) + np.eye(3)
                        print("I changed the diagonal of the bias matrix to ones.")
                    self.use_bias = use_bias.astype(np.float32)
                else:
                    raise ValueError("Bias matrix must have all positive elements.")
                self.use_bias = use_bias
            else: 
                raise ValueError("You didn't input a 3x3 bias matrix.")
        except Exception as e:
            print(f"There is a problem with your input: {e}")


def load_train_images(npz_file_path, 
                      eval_images = 100,
                      ):
    """Loader for training and evaluation images. 
    
    Loads an npz file and concatenates images and labels.  The data should be 3D
    (height,width, channels) where the greyscale image is the first channel, and
    the 2nd,3rd and 4th channels are for the boundary, interior, and exterior 
    one-hot labels.
    The evaluation images are pulled from the training images after rotating 
    them 90degrees.
    
    Args:
        npz_file_path: str filepath for npz file of image/label data
        eval_images: the number of images used for evaluation during training
    """
    npz_file = np.load(npz_file_path)   
    keys = sorted(npz_file.keys()) 
    images = np.concatenate([npz_file[key] for key in keys], axis = 0)
    np.random.shuffle(images)
    evals = images[:eval_images, ...]
    images = images[eval_images:, ...]
    images = np.concatenate([images, np.rot90(images, 1, (1, 2))], 
                             axis = 0,
                            )
    np.random.shuffle(images)
    
    image_dict={"train_images": images[..., 0:1],
                "train_labels": images[..., 1:],
                "eval_images": evals[..., 0:1],
                "eval_labels": evals[..., 1:],
                }   

    return image_dict

def conv(x,
         filters, 
         kernel_size=(3,3),
         name=None,
         params=None,
         ):
    """Convolution layers.
    
    With a fixed stride of (1,1), a conv2d layer whose input is 4D tensor 
    [batch, length, width, num of channels]
    Args: 
        x: 2d image data
        filters: number of cells/filters in layer
        kernel_size: size of 2d convolution window
        name: tf name for conv2d object
        params: params class object with regulizer and activation information
    Returns: 
        tf.layers.conv2d object.
            
    """
    return tf.layers.conv2d(x,
                            filters,
                            kernel_size,
                            strides=(1,1),
                            padding="same",
                            activation=params.activation,
                            use_bias=True,
                            kernel_regularizer=params.regularizer,
                            bias_regularizer=params.regularizer,
                            name=name)




def pool(x,
         ksize=[1, 2, 2, 1],
         strides=[1, 2, 2, 1],
         name=None,
         ):
    """Pool function layer.
    
    Args:
        x: input layer [batch, height, width, num channels]
        ksize: kernel size for pooling area
        strides: stride size for pooling window shifting
        padding: padding specification for output 
        data_format: specification for input data. Always NHWC for this 
            application
        name: tf name string
    Returns: 
        max pooling tf object
    """
    return tf.nn.max_pool(x,
                          ksize,
                          strides,
                          padding="SAME",
                          data_format='NHWC',
                          name=name)

def convT(x,
          filters,
          kernel_size=(2, 2),
          strides=(2, 2),
          name=None,
          params=None,
          ):
    """Convolution Transpose object
    
    Args: 
        x: input layer
        filters: number of filters/cells in layer
        kernel_size: size of output window 
        strides: stride size for sliding output window 
        name: tf name string
        params: params class object
    Returns:
        tf conv2d_transpose object
    """
    return tf.layers.conv2d_transpose(x,
                                      filters,
                                      kernel_size,
                                      strides = strides,
                                      padding = 'valid',
                                      data_format = 'channels_last',
                                      activation = params.activation,
                                      use_bias = True,
                                      kernel_initializer = None,
                                      bias_initializer = tf.zeros_initializer(),
                                      kernel_regularizer = params.regularizer,
                                      bias_regularizer = params.regularizer,
                                      activity_regularizer = None,
                                      kernel_constraint = None,
                                      bias_constraint = None,
                                      trainable = True,
                                      name = name,
                                      reuse = None)
                                    
def cat(x ,y ,name):
    """Concatenate x and y along last axis.
    
    Tensors must have x.shape[0:3]==y.shape[0:3] sotthey can be concatenated 
    along last axis.
    Args:
       x: 4D tensor object
       y: 4D tensor object
       name: tf name string
    """
    return tf.concat([x,y],-1,name=name)


def window_adjustment(n):
    """Adjustment list of values to trim image in "up-convolution" segments.
    
    During transpose convolution segment, if the original image had a width or
    height that was not a power of 2, then the convT will product an image size
    too big. This give a list of the adjustment for each convT layer.
    Args: 
        n: value of a image dimension (either width or height)
    Returns:
        A list of integer adjustments.
    """
    def g(x):
        return tf.ceil(x/2), tf.mod(x,2)
    l=[]
    for i in range(4):
        n,r=g(n)
        l.append(tf.cast(r,tf.int32))
    return l


        
def u_net_fn(features,labels,mode,params):
    """The u-net main function for training, evaluating and predicting masks.
    
    This is a standard u-net style network. It is meant to output masks for 
    cbjects in image data. The output masks aren't simply for object (in which
    case the output would be a boolean mask of shape (bat, height, width)), but 
    are for interior, boundary, and background. 
    
    In the params class object, weights can be stored for the custom cost  
    function below. The idea is to weight certain mislabeled ordered pairs 
    differently. For example, since boundaries may be harder to find, we might  
    not want to punish to badly a mislabeling of an exterior point as a boundary
    point. Likewise, we may wish to punish badly an interior point labeled as an
    exterior point (but perhaps visa-versa might not be as bad). The params 
    class has a 3d matrix of weights as an argument. The matrix entries for 
    (a,b) are extra penalties for 'a' being mislabeled as 'b'. Lastly, the 
    penalities are the exp**(penality). If there is no extra penalty desired for
    a particular mislabeling, then that value should show up as 1 in the matrix.
    To clarify how this particular penalty is applied, it shows up as a weight
    in the categorical cross entropy loss function, but through adding to the 
    predicted logit output a value that is ln of matrix value for the type of 
    mislabeling. Thus after the cross entropy loss, it shows up as a multiplier
    on the type of mislabeling. 
    
    Also in the loss function is an option to give more loss to not only the 
    boundary but to a 'fat boundary' which is set of all pixels touching the 
    boundary. This is also set in params
    Args:
        features: dict {"x": image batch}
        labels: expected image output shape = (bat,height,width,3)
        mode: tf estimator mode
        params: params class object
    Returns:
        A tf.estimator.EstimatorSpec object of type depending on mode.
    """
    
    c_00 = features["x"]
    bat = tf.shape(c_00)[0]
    
    v_adj = window_adjustment(tf.shape(c_00)[1])
    h_adj = window_adjustment(tf.shape(c_00)[2])
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True
    else:
        training = False
    #0th level of convolutions
    c_01=conv(c_00, params.layers[0], name="conv01_down", params=params)
    c_01=tf.layers.dropout(c_01, 
                           rate=params.rates[0], 
                           noise_shape=[bat, 1, 1, params.layers[0]],
                           training =training,
                           )
    c_02=conv(c_01, params.layers[0], name="conv02_down", params = params)
    c_02=tf.layers.dropout(c_02,
                           rate = params.rates[0],
                           noise_shape = [bat, 1, 1, params.layers[0]],
                           training = training,
                           )
    c_10=pool(c_02, name = "pool10")
    #1st level down
    c_11=conv(c_10, params.layers[1], name = "conv11_down", params = params)
    c_11=tf.layers.dropout(c_11,
                           rate = params.rates[1],
                           noise_shape = [bat, 1, 1, params.layers[1]],
                           training = training)
    c_12=conv(c_11, params.layers[1], name = "conv12_down", params = params)
    c_12=tf.layers.dropout(c_12,
                           rate = params.rates[1],
                           noise_shape = [bat, 1, 1, params.layers[1]],
                           training = training)
    c_20=pool(c_12, name = "pool20")
    #2nd level down
    c_21=conv(c_20, params.layers[2], name = "conv21_down", params = params)
    c_21=tf.layers.dropout(c_21,
                           rate = params.rates[2],
                           noise_shape = [bat, 1, 1, params.layers[2]],
                           training = training,
                           )
    c_22=conv(c_21,params.layers[2], name = "conv22_down", params = params)
    c_22=tf.layers.dropout(c_22,
                           rate = params.rates[2],
                           noise_shape = [bat, 1, 1, params.layers[2]],
                           training =training,
                           )
    c_30 = pool(c_22, name="pool30")
    #3rd level down
    c_31 = conv(c_30, params.layers[3], name = "conv31_down", params = params)
    c_31 = tf.layers.dropout(c_31,
                             rate = params.rates[3],
                             noise_shape = [bat,1,1,params.layers[3]],
                             training = training,
                             )
    c_32 = conv(c_31,params.layers[3], name = "conv32_down", params = params)
    c_32 = tf.layers.dropout(c_32,
                             rate = params.rates[3],
                             noise_shape = [bat, 1, 1, params.layers[3]],
                             training = training,
                             )
    c_40 = pool(c_32, name = "pool40")
    #4th level down
    c_41 = conv(c_40, params.layers[4], name = "conv41_down", params = params)
    c_41 = tf.layers.dropout(c_41,
                             rate = params.rates[4],
                             noise_shape = [bat, 1, 1, params.layers[4]],
                             training = training,
                             )
    c_42 = conv(c_41, params.layers[4], name = "conv42_down", params = params)
    c_42 = tf.layers.dropout(c_42,
                             rate = params.rates[4],
                             noise_shape = [bat, 1, 1, params.layers[4]],
                             training = training,
                           )
    cT_30 = convT(c_42, params.layers[3], name = "convT30_up", params = params)
    #3rd level up
    v = tf.shape(cT_30)[1] - v_adj[3]
    h = tf.shape(cT_30)[2] - h_adj[3]
    c_31 = conv(cat(cT_30[:,:v,:h,:], c_32, name = "cat31_up"), 
                params.layers[3], 
                name = "conv31_up",
                params = params,
                )
    c_31 = tf.layers.dropout(c_31,
                             rate = params.rates[5],
                             noise_shape = [bat, 1, 1, params.layers[3]],
                             training = training,
                             )
    c_32 = conv(c_31, params.layers[3], name = "conv32_up", params = params)
    c_32 = tf.layers.dropout(c_32,
                             rate = params.rates[5],
                             noise_shape = [bat, 1, 1, params.layers[3]],
                             training = training,
                             )
    cT_20 = convT(c_32, params.layers[2], name = "convT20_up", params = params)
    #2nd level up
    v = tf.shape(cT_20)[1] - v_adj[2]
    h = tf.shape(cT_20)[2] - h_adj[2]
    c_21 = conv(cat(cT_20[:,:v,:h,:],c_22, name = "cat21_up"),
                params.layers[2], 
                name = "conv21_up",
                params = params,
                )
    c_21 = tf.layers.dropout(c_21,
                             rate = params.rates[6],
                             noise_shape = [bat, 1, 1, params.layers[2]],
                             training = training)
    c_22 = conv(c_21, params.layers[2], name = "conv22_up", params = params)
    c_22 = tf.layers.dropout(c_22,
                             rate = params.rates[6],
                             noise_shape = [bat, 1, 1, params.layers[2]],
                             training = training,
                             )
    cT_10 = convT(c_22, params.layers[1], name = "convT10_up", params = params)
    #1st level up
    v = tf.shape(cT_10)[1] - v_adj[1]
    h = tf.shape(cT_10)[2] - h_adj[1]
    c_11 = conv(cat(cT_10[:,:v,:h,:], c_12, name = "cat11_up"),
                params.layers[1], 
                name = "conv11_up",
                params = params,
                )
    c_11 = tf.layers.dropout(c_11,
                             rate=params.rates[7],
                             noise_shape=[bat,1,1,params.layers[1]],
                             training=training,
                             )
    c_12 = conv(c_11,params.layers[1],
                name = "conv12_up",
                params = params),
                
    c_12 = tf.layers.dropout(c_12,
                             rate = params.rates[7],
                             noise_shape = [bat, 1, 1, params.layers[1]],
                             training = training,
                             )
    
    cT_00 = convT(c_12, params.layers[0], name="convT10_up", params = params)
    #0th level up
    v = tf.shape(cT_00)[1] - v_adj[0]
    h = tf.shape(cT_00)[2] - h_adj[0]
    c_01 = conv(cat(cT_00[:,:v,:h,:], c_02, name = "cat01_up"),
                params.layers[0], 
                name = "conv01_up",
                params = params,
                )
    c_01 = tf.layers.dropout(c_01,
                             rate=params.rates[8],
                             noise_shape = [bat, 1, 1, params.layers[0]],
                             training = training,
                             )
    c_02 = conv(c_01, params.layers[0], name = "conv02_up", params = params)
    c_02 = tf.layers.dropout(c_02,
                             rate = params.rates[8],
                             noise_shape = [bat, 1, 1, params.layers[0]],
                             training = training,
                             )
    #end. concatonate with original image
    logit_params = copy.copy(params)
    logit_params.activation = None
    logits = conv(cat(c_02, c_00, name= "last_image_concat"),
                  3,
                  kernel_size = (1,1),
                  name = "logits_out",
                  params = logit_params,
                  )
   
    def loss_with_parameters(y_true,y_pred):
        """Loss function for model.
        
        Args: 
            y_true: expected labels
            y_pred: predicted labels
        Returns:
            weighted loss
        """
        try:
            params.use_bias.shape == (3,3)
            bias = tf.convert_to_tensor(params.use_bias)
            ln = tf.log(bias)
            w = tf.einsum("...b,ba->...a", y_true, ln)
            logits = tf.add(w, y_pred)
        except:
            logits = y_pred
            
        bdy_cost = params.bdy_cost
        int_cost = params.int_cost
        bgd_cost = params.bgd_cost
        fat_bdy_on = params.fat_bdy_on
        weights_notbdy = tf.convert_to_tensor([bgd_cost, int_cost],
                                              dtype = tf.float32,
                                              )
        
        weighted_notbdy = tf.reduce_sum(tf.multiply(weights_notbdy,
                                                    y_true[:, :, :, 0:2],
                                                    ),
                                        axis = -1,
                                        keepdims = True,
                                        )
        
        boundary = y_true[:, :, :, 2:3]
        
        if fat_bdy_on:
            fat_bdy = tf.nn.max_pool(boundary,
                                     ksize = [1, 3, 3, 1],
                                     strides = [1, 1, 1, 1],
                                     padding = "SAME",
                                     data_format = "NHWC",
                                     )
        else:
            fat_bdy = boundary
            
        weighted_bdy = tf.scalar_mul(bdy_cost, fat_bdy)
        intersect = tf.multiply(fat_bdy, weighted_notbdy)
        weights = tf.add(weighted_bdy, weighted_notbdy)
        weights = tf.subtract(weights,intersect)
        cost = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_true,
                                                          logits = logits,
                                                          )
        cost = tf.expand_dims(cost, -1)
        cost = tf.multiply(weights, cost)
        return tf.reduce_mean(cost)
    
    predictions = {"probabilities": tf.nn.softmax(logits,
                                                  name = "softmax_tensor",
                                                  ),
                   "labels": tf.one_hot(tf.argmax(logits, axis = -1),
                                        3,
                                        axis = -1,
                                        name = "output_labels"),
                }
    
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)


    
    loss = loss_with_parameters(labels, logits)
    if params.regularizer != None:
        reg_loss = tf.losses.get_regularization_losses()
        loss = loss + tf.reduce_sum(reg_loss)
        
    
    eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels = tf.argmax(labels,
                                                                           axis = -1,
                                                                           ),
                                                        predictions = tf.argmax(predictions["labels"], 
                                                                                axis = -1,
                                                                                ),
                                                        name = "accuracy_1"),
                    }
    
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions["labels"],
                                                    axis = -1,
                                                    ),
                                          tf.argmax(labels, axis = -1)),
                                          tf.float32,
                                          ),
                        name="accuracy_2",
                        )
    pred = predictions["labels"]
    bgd = tf.divide(tf.reduce_sum(tf.multiply(pred[:,:,:,0:1],
                                              labels[:,:,:,0:1]),
                                              ),
                    tf.reduce_sum(labels[:,:,:,0:1]),
                    name = "bgd_true_pos",
                    )
    intr = tf.divide(tf.reduce_sum(tf.multiply(pred[:,:,:,1:2],
                                               labels[:,:,:,1:2]),
                                               ),
                     tf.reduce_sum(labels[:,:,:,1:2]),
                     name = "int_true_pos",
                    )
    bdy = tf.divide(tf.reduce_sum(tf.multiply(pred[:,:,:,2:3],
                                              labels[:,:,:,2:3]),
                                              ),
                    tf.reduce_sum(labels[:,:,:,2:3]),
                    name = "bdy_true_pos",
                    )
    
    tf.summary.scalar("pixel_wise_average_accuracy", acc)
    tf.summary.scalar("weighted_loss", loss)
    tf.summary.scalar("background_true_pos", bgd)
    tf.summary.scalar("interior_true_pos", intr)
    tf.summary.scalar("boundary_true_pos", bdy)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = params.learning_rate,
                                           epsilon = params.epsilon,
                                           )
        grads, varis = [*zip(*optimizer.compute_gradients(loss = loss))]
        clip_grads = [tf.clip_by_average_norm(grad, params.clip_norm) for grad in grads]
        
        train_op = optimizer.apply_gradients([*zip(clip_grads,varis)],
                                              global_step = tf.train.get_global_step(),
                                            )
        
        return tf.estimator.EstimatorSpec(mode = mode,
                                          loss = loss,
                                          train_op = train_op,
                                         )
        
    return tf.estimator.EstimatorSpec(mode = mode,
                                      loss = loss,
                                      eval_metric_ops = eval_metric_ops,
                                      )
   
def nuclei_u_net(model_dir,image_data,params):
    """The training, evaluating, and predicting function for u_net_fn.
    
    Args:
        model_dir: the directory for finding/storing model
        image_data: a dictionary of image/label data with keys "train_images"
                    "train_labels", "eval_images", "eval_labels", 
                    "predict_images". Can be used with only predict or only 
                    train, or eval entries. 
        params: a hyperparams class object for training parameters
    """
    
    my_config = tf.estimator.RunConfig(save_checkpoints_steps = params.save_checkpoints_steps,
                                       keep_checkpoint_max = params.keep_checkpoint_max,
                                       save_summary_steps = params.save_summary_steps,
                                       )
    u_net = tf.estimator.Estimator(model_fn = u_net_fn,
                                   model_dir = model_dir,
                                   config = my_config,
                                   params = params,
                                   )   
    if "train_images" in image_data.keys():
        tensors_to_log = {"pixel_wise_ave_accuracy": "accuracy_2",
                          "bgd_true_pos": "bgd_true_pos",
                          "int_true_pos": "int_true_pos",
                          "bdy_true_pos": "bdy_true_pos",
                          }
        logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log,
                                                  every_n_iter = params.logging_step,
                                                  )
        
       
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"x": image_data["train_images"]},
                                                            y = image_data["train_labels"],
                                                            batch_size = params.batch_size,
                                                            num_epochs = None,
                                                            shuffle = True)
        u_net.train(input_fn = train_input_fn,
                    steps = params.training_steps,
                    hooks = [logging_hook],
                    )
        
    if "eval_images" in image_data.keys():
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"x": image_data["eval_images"]},
                                                           y = image_data["eval_labels"],
                                                           num_epochs = 1,
                                                           shuffle = False,
                                                           )
        eval_results = u_net.evaluate(input_fn = eval_input_fn,
                                      checkpoint_path = params.checkpoint_path,
                                      )
        print(eval_results)
       
    if "predict_images" in image_data.keys():
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x = {"x": image_data["predict_images"]},
                                                              shuffle = False,
                                                              batch_size = params.predict_batch_size,
                                                              num_epochs = 1,
                                                              )
        return [x for x in u_net.predict(input_fn=predict_input_fn,
                                         checkpoint_path=params.checkpoint_path)]