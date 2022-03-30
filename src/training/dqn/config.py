import tensorflow as tf
from tf_agents.utils import common

class config:
    seed = 0

    # networks
    # refer to https://www.tensorflow.org/agents/api_docs/python/tf_agents/networks/q_network/QNetwork
    use_ddqn = True
    fc_layer_params = (150, 150)
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
    gamma = 0.99
    target_update_tau = 1.0
    target_update_period = 100
    td_errors_loss_fn = common.element_wise_squared_loss

    # data collection
    initial_collect_epispode = 100
    replay_buffer_max_length = 20000
    collect_steps_per_epoch = 10

    # loggings
    log_iterval = 200
    ckpt_interval = 500
    ckpt_kept_num = 5

    # validation
    valid_data_dir = "../testdata/test-2x2-s/"
    valid_interval = 500
