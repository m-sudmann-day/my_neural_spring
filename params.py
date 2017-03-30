from enum import Enum
import copy as cp

class data_source(Enum):
    PIPES = 1
    CIFAR10 = 2
    MNIST = 3
    SYNTH = 4

class execution_mode(Enum):
    TRAINING = 1
    VALIDATION = 2
    DECONV_VISUALIZATION = 3
    GUIDED_BACKPROP_VISUALIZATION = 4

class batch_type(Enum):
    MINI = 1
    SHUFFLED_MINI = 2
    SAMPLE_WITH_REPLACEMENT = 3
    SAMPLE_WITHOUT_REPLACEMENT = 4

class batch_normalization_method(Enum):
    NONE = 1
    TENSORFLOW_BUILTIN = 2
    TRAINABLE = 3

class batch_normalization_timing(Enum):
    NONE = 1
    BEFORE_NONLINEARITY = 2 # typical
    AFTER_NONLINEARITY = 3 # I read that this is better when dropout is used.  Need to test it.

# See https://openreview.net/pdf?id=r1BJLw9ex
# TODO Do I need to adjust weight initialization to compensate for dropped out neurons?

class weight_initialization_method(Enum):
    RANDOM_NORMAL = 1
    HE_ET_AL = 2

class bias_initialization_method(Enum):
    RANDOM_NORMAL = 1
    SMALL_NUMBER = 2

class pooling_method(Enum):
    MAX = 1
    MEAN = 2

class activation_function(Enum):
    RELU = 1
    RELU_TENSORFLOW_BUILTIN = 2
    XELU = 3
    PRELU = 4
    SIGMOID = 5 # not supported
    TANH = 6 # not supported

class loss_function(Enum):
    MULTICLASS_SVM_L1_HINGE = 1
    MULTICLASS_SVM_L2_HINGE = 2
    SOFTMAX_CROSS_ENTROPY = 3

class regularization_penalty(Enum):
    NONE = 1
    L1 = 2
    L2 = 3
    L2_WITH_DECAY = 4

class optimization_method(Enum):
    ADAM = 1
    ADAGRAD = 2
    # waiting for Nesterov Momentum to be added to Adam

class msd_params:

    execution_mode = None
    data_source = None
    dropout_ratio_conv = None
    dropout_ratio_fc = None
    train_ratio = None
    max_inputs = None
    num_epochs = None
    learning_rate = None
    batch_size = None
    batch_type = None
    batch_normalization_method = None
    activation_function = None
    loss_function = None
    regularization_penalty = None
    regularization_strength = None
    optimization_method = None
    shuffle_data = None
    shuffle_data_seed = None
    batches_per_super_batch = None

    def copy(self):
        return cp.copy(self)
        