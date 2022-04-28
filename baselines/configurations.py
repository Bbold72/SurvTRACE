from easydict import EasyDict

################################## Globals ##################################
horizons = [.25, .5, .75]
batch_size_metabric = 64
batch_size_support = 128
batch_size_seer = 1024

########################## Cox Proportional Hazards #########################

### Metabric ###
CPH_metabric = EasyDict({
    'data': 'metabric',
    'horizons': horizons,
    'epochs': 200
})

### SUPPORT ###
CPH_support = EasyDict({
    'data': 'support',
    'horizons': horizons,
    'epochs': 200
})
### SEER ###
CPH_seer = EasyDict({
    'data': 'seer',
    'horizons': horizons,
    'epochs': 200,
})


########################## DeepHit ##########################

### Metabric ###
DeepHit_metabric = EasyDict({
    'data': 'metabric',
    'horizons': horizons,
    'batch_size': batch_size_metabric,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SUPPORT ###
DeepHit_support = EasyDict({
    'data': 'support',
    'horizons': horizons,
    'batch_size': batch_size_support,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SEER ###
DeepHit_seer = EasyDict({
    'data': 'seer',
    'horizons': horizons,
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
    'data': 'metabric',
    'horizons': horizons,
    'batch_size': batch_size_metabric,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SUPPORT ###
DeepSurv_support = EasyDict({
    'data': 'support',
    'horizons': horizons,
    'batch_size': batch_size_support,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SEER ###
DeepSurv_seer = EasyDict({
    'data': 'seer',
    'horizons': horizons,
    'batch_size': batch_size_support,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1,
})


########################## Deep Survival Machines ##########################

### Metabric ###
DSM_metabric = EasyDict({
    'data': 'metabric',
    'horizons': horizons,
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
    'data': 'support',
    'horizons': horizons,
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
    'data': 'seer',
    'horizons': horizons,
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
    'data': 'metabric',
    'horizons': horizons,
    'batch_size': batch_size_metabric,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SUPPORT ###
PCHazard_support = EasyDict({
    'data': 'support',
    'horizons': horizons,
    'batch_size': batch_size_support,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SEER ###
PCHazard_seer = EasyDict({
    'data': 'seer',
    'horizons': horizons,
    'batch_size': batch_size_seer,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1,
})


########################## Random Survival Forests ##########################

### Metabric ###
RSF_metabric = EasyDict({
    'data': 'metabric',
    'horizons': horizons,
    'epochs': 100
})

### SUPPORT ###
RSF_support = EasyDict({
    'data': 'support',
    'horizons': horizons,
    'epochs': 100
})

### SEER ###
RSF_seer = EasyDict({
    'data': 'seer',
    'horizons': horizons,
    'epochs': 100,
})