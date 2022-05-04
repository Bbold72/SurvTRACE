from easydict import EasyDict

# defines various parameters for each experiment and model.
# Note: hyperparameters depend on each model and dataset
# naming of each dictionary: '{model name}_{dataset name}'.

# TODO: add brief definitions to paramaters
# TODO: maybe this can be reduced by using classes per model

################################## Globals ##################################
batch_size_metabric = 64
batch_size_support = 128
batch_size_seer = 1024

########################## Cox Proportional Hazards #########################

### Metabric ###
CPH_metabric = EasyDict({
    'epochs': 100
})

### SUPPORT ###
CPH_support = EasyDict({
    'epochs': 100
})
### SEER ###
CPH_seer = EasyDict({
    'epochs': 100,
})


########################## DeepHit ##########################
# alpha: governs the contribution of likelihoo and rank loss
# sigma: scales loss function

### Metabric ###
DeepHit_metabric = EasyDict({
    'batch_size': batch_size_metabric,
    'epochs': 100,

    # network
    'hidden_layers_size': [16],
    'dropout': 0.1,

    # loss
    'alpha': 0.2,
    'sigma': 0.1,

    # optimizer
    'learning_rate': 0.01
})

### SUPPORT ###
DeepHit_support = EasyDict({
    'batch_size': batch_size_support,
    'epochs': 100,

    # network
    'hidden_layers_size': [16, 16],
    'dropout': 0.1,

    # loss
    'alpha': 0.2,
    'sigma': 0.1,

    # optimizer
    'learning_rate': 0.01
})

### SEER ###
DeepHit_seer = EasyDict({
    'batch_size': batch_size_seer,
    'epochs': 100,

    # network
    'hidden_layers_size': [32, 32],
    'dropout': 0.1,

    # loss
    'alpha': 0.2,
    'sigma': 0.1,

    # optimizer
    'learning_rate': 0.001
})


########################## DeepSurv ##########################

### Metabric ###
DeepSurv_metabric = EasyDict({
    'batch_size': batch_size_metabric,
    'epochs': 100,

    # Network
    'hidden_layers_size': [32, 32],
    'dropout': 0.1,

    # Adam
    'learning_rate': 1e-2,
    'weight_decay': 0,
})

### SUPPORT ###
DeepSurv_support = EasyDict({
    'batch_size': batch_size_support,
    'epochs': 100,

    # Network
    'hidden_layers_size': [32, 32],
    'dropout': 0.1,

    # Adam
    'learning_rate': 1e-2,
    'weight_decay': 0,
})

### SEER ###
DeepSurv_seer = EasyDict({
    'batch_size': batch_size_seer,
    'epochs': 100,

    # Network
    'hidden_layers_size': [64, 64],
    'dropout': 0.1,

    # Adam
    'learning_rate': 1e-2,
    'weight_decay': 0.1,
})


########################## Deep Survival Machines ##########################
# hidden_size (list): number of layers and number of nodex
# k: number of underlying parametric distributions

### Metabric ###
DSM_metabric = EasyDict({
    'batch_size': batch_size_metabric,
    'epochs': 100,
    'hidden_size': [100, 100],
    'k': 4,
    'distribution': 'Weibull',
    'learning_rate': 1e-3,
    'discount': 0.5
    })


### SUPPORT ###
DSM_support = EasyDict({
    'batch_size': batch_size_support,
    'epochs': 100,
    'hidden_size': [100, 100],
    'k': 4,
    'distribution': 'Weibull',
    'learning_rate': 1e-3,
    'discount': 0.5
    })


### SEER ###
DSM_seer = EasyDict({
    'batch_size': batch_size_seer,
    'epochs': 100,
    'hidden_size': [100, 100],
    'k': 4,
    'distribution': 'Weibull',
    'learning_rate': 1e-4,
    'discount': 0.5
    })



########################## PC-Hazard ##########################

### Metabric ###
PCHazard_metabric = EasyDict({
    'batch_size': batch_size_metabric,
    'epochs': 100,

    # Network
    'hidden_layers_size': [64, 64],
    'dropout': 0.1,

    # AdamWR
    'learning_rate': 1e-2,
    'decoupled_weight_decay': 0.8,
    'cycle_multiplier': 2
})


### SUPPORT ###
PCHazard_support = EasyDict({
    'batch_size': batch_size_support,
    'epochs': 100,

    # Network
    'hidden_layers_size': [32, 32],
    'dropout': 0.1,

    # AdamWR
    'learning_rate': 1e-2,
    'decoupled_weight_decay': 0.8,
    'cycle_multiplier': 2
})

### SEER ###
PCHazard_seer = EasyDict({
    'batch_size': batch_size_seer,
    'epochs': 100,

    # Network
    'hidden_layers_size': [32, 32, 32, 32],
    'dropout': 0.1,

    # AdamWR
    'learning_rate': 1e-3,
    'decoupled_weight_decay': 0.8,
    'cycle_multiplier': 2
})


########################## Random Survival Forests ##########################
# epoch: in this context, refers to number of trees to generate in the forest.
# max_depth: maximum depth of the tree.

### Metabric ###
RSF_metabric = EasyDict({
    'epochs': 200,
    'max_depth': 4
})

### SUPPORT ###
RSF_support = EasyDict({
    'epochs': 200,
    'max_depth': 4
})

### SEER ###
RSF_seer = EasyDict({
    'epochs': 200,
    'max_depth': 4
})

########################## SurvTRACE #########################
# Different configs for SurvTRACE only based on differing datasets
#   and not on different variants.

### METABRIC ###
survtrace_metabric = EasyDict(
    {
        'num_durations': 5, # num of discrete intervals for prediction, e.g., num_dur = 5 means the whole period is discretized to be 5 intervals
        'seed': 1234,
        'checkpoint': './checkpoints/survtrace.pt',
        'vocab_size': 8, # num of all possible values of categorical features
        'hidden_size': 16, # embedding size
        'intermediate_size': 64, # intermediate layer size in transformer layer
        'num_hidden_layers': 3, # num of transformers
        'num_attention_heads': 2, # num of attention heads in transformer layer
        'hidden_dropout_prob': 0.0,
        'num_feature': 9, # num of covariates of patients, should be set during load_data
        'num_numerical_feature': 5, # num of numerical covariates of patients, should be set during load_data
        'num_categorical_feature': 4, # num of categorical covariates of patients, should be set during load_data
        'out_feature':3, # equals to the length of 'horizons', indicating the output dim of the logit layer of survtrace
        'num_event': 1, # only set when using SurvTraceMulti for competing risks
        'hidden_act': 'gelu',
        'attention_probs_dropout_prob': 0.1,
        'early_stop_patience': 10,
        'initializer_range': 0.001,
        'layer_norm_eps': 1e-12,
        'max_position_embeddings': 512, # # no use
        'chunk_size_feed_forward': 0, # no use
        'output_attentions': False, # no use
        'output_hidden_states': False, # no use 
        'tie_word_embeddings': True, # no use
        'pruned_heads': {}, # no use

        # hyperparameters
        'batch_size': batch_size_metabric,
        'weight_decay': 1e-4,
        'learning_rate': 1e-3,
        'epochs': 100
    }
)

### SUPPORT ###
survtrace_support = EasyDict(
    {
        'num_durations': 5, # num of discrete intervals for prediction, e.g., num_dur = 5 means the whole period is discretized to be 5 intervals
        'seed': 1234,
        'checkpoint': './checkpoints/survtrace.pt',
        'vocab_size': 8, # num of all possible values of categorical features
        'hidden_size': 16, # embedding size
        'intermediate_size': 64, # intermediate layer size in transformer layer
        'num_hidden_layers': 3, # num of transformers
        'num_attention_heads': 2, # num of attention heads in transformer layer
        'hidden_dropout_prob': 0.0,
        'num_feature': 9, # num of covariates of patients, should be set during load_data
        'num_numerical_feature': 5, # num of numerical covariates of patients, should be set during load_data
        'num_categorical_feature': 4, # num of categorical covariates of patients, should be set during load_data
        'out_feature':3, # equals to the length of 'horizons', indicating the output dim of the logit layer of survtrace
        'num_event': 1, # only set when using SurvTraceMulti for competing risks
        'hidden_act': 'gelu',
        'attention_probs_dropout_prob': 0.1,
        'early_stop_patience': 10,
        'initializer_range': 0.001,
        'layer_norm_eps': 1e-12,
        'max_position_embeddings': 512, # # no use
        'chunk_size_feed_forward': 0, # no use
        'output_attentions': False, # no use
        'output_hidden_states': False, # no use 
        'tie_word_embeddings': True, # no use
        'pruned_heads': {}, # no use

        # hyperparameters
        'batch_size': batch_size_support,
        'weight_decay': 0,
        'learning_rate': 1e-3,
        'epochs': 100
    }
)


### SEER ###
survtrace_seer = EasyDict(
    {
        'num_durations': 5, # num of discrete intervals for prediction, e.g., num_dur = 5 means the whole period is discretized to be 5 intervals
        'seed': 1234,
        'checkpoint': './checkpoints/survtrace.pt',
        'vocab_size': 8, # num of all possible values of categorical features
        'hidden_size': 16, # embedding size
        'intermediate_size': 64, # intermediate layer size in transformer layer
        'num_hidden_layers': 2, # num of transformers
        'num_attention_heads': 2, # num of attention heads in transformer layer
        'hidden_dropout_prob': 0.0,
        'num_feature': 9, # num of covariates of patients, should be set during load_data
        'num_numerical_feature': 5, # num of numerical covariates of patients, should be set during load_data
        'num_categorical_feature': 4, # num of categorical covariates of patients, should be set during load_data
        'out_feature':3, # equals to the length of 'horizons', indicating the output dim of the logit layer of survtrace
        'num_event': 1, # only set when using SurvTraceMulti for competing risks
        'hidden_act': 'gelu',
        'attention_probs_dropout_prob': 0.1,
        'early_stop_patience': 5,
        'initializer_range': 0.02,
        'layer_norm_eps': 1e-12,
        'max_position_embeddings': 512, # # no use
        'chunk_size_feed_forward': 0, # no use
        'output_attentions': False, # no use
        'output_hidden_states': False, # no use 
        'tie_word_embeddings': True, # no use
        'pruned_heads': {}, # no use
        'val_batch_size': 10000,

        # hyperparameters
        'batch_size': batch_size_seer,
        'weight_decay': 0,
        'learning_rate': 1e-4,
        'epochs': 100
    }
)