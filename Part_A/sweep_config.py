sweep_config = {
    "method": "bayes",  # or "grid", "random"
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "conv_filters": {
            "values": [
                [32, 32, 32, 32, 32],         # same
                [32, 64, 128, 256, 512],      # doubling
                [512, 256, 128, 64, 32],      # halving
                [64, 128, 128, 256, 256]      # custom
            ]
        },
        "kernel_sizes": {
            "value": [3, 3, 3, 3, 3]
        },
        "activation": {
            "values": ["relu", "gelu", "silu", "mish"]
        },
        "batch_norm": {
            "values": [True, False]
        },
        "dropout": {
            "values": [
                [0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3]
            ]
        },
        "dense_neurons": {
            "values": [256, 512, 1024]
        },
        "learning_rate": {
            "min": 0.0001,
            "max": 0.01
        },
        "batch_size": {
            "values": [32, 64, 128]
        },
        "epochs": {
            "value": 10
        },
        "optimizer": {
            "values": ["adam", "sgd", "rmsprop"]
        },
        "data_augmentation": {
            "values": [True, False]
        }
    }
}
