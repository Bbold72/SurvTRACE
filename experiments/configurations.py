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
    'epochs': 200
})

### SUPPORT ###
CPH_support = EasyDict({
    'epochs': 200
})
### SEER ###
CPH_seer = EasyDict({
    'epochs': 200,
})


########################## DeepHit ##########################

### Metabric ###
DeepHit_metabric = EasyDict({
    'batch_size': batch_size_metabric,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SUPPORT ###
DeepHit_support = EasyDict({
    'batch_size': batch_size_support,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SEER ###
DeepHit_seer = EasyDict({
    'batch_size': batch_size_seer,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size_indiv': 32,
    'hidden_size_shared': 64,
    'dropout': 0.1
})


########################## DeepSurv ##########################

### Metabric ###
DeepSurv_metabric = EasyDict({
    'batch_size': batch_size_metabric,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SUPPORT ###
DeepSurv_support = EasyDict({
    'batch_size': batch_size_support,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SEER ###
DeepSurv_seer = EasyDict({
    'batch_size': batch_size_support,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1,
})


########################## Deep Survival Machines ##########################

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
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SUPPORT ###
PCHazard_support = EasyDict({
    'batch_size': batch_size_support,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SEER ###
PCHazard_seer = EasyDict({
    'batch_size': batch_size_seer,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1,
})


########################## Random Survival Forests ##########################

### Metabric ###
RSF_metabric = EasyDict({
    'epochs': 100
})

### SUPPORT ###
RSF_support = EasyDict({
    'epochs': 100
})

### SEER ###
RSF_seer = EasyDict({
    'epochs': 100,
})

########################## SurvTRACE #########################
# Different configs for SurvTRACE only based on differing datasets
#   and not on different variants.

### METABRIC ###
SurvTRACE_metabric = EasyDict(
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
        'batch_size': 64,
        'weight_decay': 1e-4,
        'learning_rate': 1e-3,
        'epochs': 100
    }
)

### SUPPORT ###
SurvTRACE_support = EasyDict(
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
        'batch_size': 128,
        'weight_decay': 0,
        'learning_rate': 1e-3,
        'epochs': 100
    }
)


### SUPPORT ###
SurvTRACE_seer = EasyDict(
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
        'batch_size': 1024,
        'weight_decay': 0,
        'learning_rate': 1e-4,
        'epochs': 100
    }
)