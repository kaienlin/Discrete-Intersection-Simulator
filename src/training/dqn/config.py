import tensorflow as tf
from tf_agents.utils import common

class config:
    seed = 0

    # networks
    # refer to https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/q_network/QNetwork
    fc_layer_params = (75, 40)
    dropout_layer_params = (0.1, 0.1)
    conv_layer_params = None
    activation_fn = tf.keras.activations.relu
    kernel_initializer = None

    # training
    is_cuda = True
    num_iterations = 100000
    batch_size = 1
    lr = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-6
    td_errors_loss_fn = common.element_wise_squared_loss

    # data collection
    replay_buffer_max_length = 20000
    collect_steps_per_epoch = 10

    # loggings
    log_iterval = 200
    log_dir = 'loggings'