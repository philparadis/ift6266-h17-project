default_test_hyper_params = {
    "input_dim" : 64*64*3 - 32*32*3,
    "output_dim" : 32*32*3,
    "loss_function" : "mse",
    "optimizer" : "adam",
    "learning_rate" : 2e-4,
    "batch_size" : 32
    }
                            

default_mlp_hyper_params = {
    "input_dim" : 64*64*3 - 32*32*3,
    "output_dim" : 32*32*3,
    "loss_function" : "mse",
    "optimizer" : "adam",
    "learning_rate" : 1e-4,
    "batch_size" : 64
    }

default_conv_mlp_hyper_params = {
    "input_dim" : (None, 3, 64, 64),
    "output_dim" : 32*32*3,
    "loss_function" : "mse",
    "optimizer" : "adam",
    "learning_rate" : 1e-5,
    "batch_size" : 128,
    "convolution" : { "receptive_field_size" : 5, "stride" : 1, "padding" : "same" }
    }

default_dcgan_hyper_params = {
    "input_dim" : (None, 3, 64, 64),
    "output_dim" : (None, 3, 64, 64),
    "loss_function" : "minmax",
    "optimizer" : "adam",
    "initial_learning_rate" : 2e-4,
    "batch_size" : 128
    }

default_wgan_hyper_params = {
    "input_dim" : (None, 3, 64, 64),
    "output_dim" : (None, 3, 64, 64),
    "loss_function" : "minmax",
    "optimizer" : "adam",
    "learning_rate" : 5e-5,
    "batch_size" : 64
    }

default_lsgan_hyper_params = {
    "input_dim" : (None, 3, 64, 64),
    "output_dim" : (None, 3, 64, 64),
    "loss_function" : "minmax",
    "optimizer" : "adam",
    "learning_rate" : 1e-4,
    "batch_size" : 64
    }
