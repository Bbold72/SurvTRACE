from easydict import EasyDict


########################## Cox Proportional Hazards ##########################

### Metabric ###
CPH_metabric = EasyDict({
    'data': 'metabric',
    'horizons': [.25, .5, .75],
    'epochs': 200
})

### SUPPORT ###
CPH_support = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'epochs': 200
})
### SEER ###
CPH_seer = EasyDict({
    'data': 'seer',
    'horizons': [.25, .5, .75],
    'epochs': 200,
})


########################## DeepHit ##########################

### Metabric ###
DeepHit_metabric = EasyDict({
    'data': 'metabric',
    'horizons': [.25, .5, .75],
    'batch_size': 64,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SUPPORT ###
DeepHit_support = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'batch_size': 128,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

### SEER ###
DeepHit_seer = EasyDict({
    'data': 'seer',
    'horizons': [.25, .5, .75],
    'batch_size': 1024,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size_indiv': 32,
    'hidden_size_shared': 64,
    'dropout': 0.1
})


########################## Deep Survival Machines ##########################

### Metabric ###
DSM_metabric = EasyDict({
    'data': 'metabric',
    'horizons': [.25, .5, .75],
    'batch_size': 64,
    'epochs': 100,
    'hyperparameters': EasyDict({
        'hidden_size': [[100, 100]],
        'k': [4],
        'distribution': ['Weibull'],
        'learning_rate': [1e-3],
        'discount': [0.5]
        # 'hidden_size': [[50], [50, 50], [100], [100, 100]],
        # 'k': [4, 6, 8],
        # 'distribution': ['LogNormal', 'Weibull'],
        # 'learning_rate': [1e-3, 1e-4],
        # 'discount': [0.5, 0.75, 1]
    })
})

### SUPPORT ###
DSM_support = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'batch_size': 128,
    'epochs': 100,
    'hyperparameters': EasyDict({
        'hidden_size': [[100, 100]],
        'k': [4],
        'distribution': ['Weibull'],
        'learning_rate': [1e-3],
        'discount': [0.5]
        # 'hidden_size': [[50], [50, 50], [100], [100, 100]],
        # 'k': [4, 6, 8],
        # 'distribution': ['LogNormal', 'Weibull'],
        # 'learning_rate': [1e-3, 1e-4],
        # 'discount': [0.5, 0.75, 1]
    })
})

### SEER ###
DSM_seer = EasyDict({
    'data': 'seer',
    'horizons': [.25, .5, .75],
    'batch_size': 1024,
    'epochs': 100,
    'hyperparameters': EasyDict({
        'hidden_size': [[100, 100]],
        'k': [4],
        'distribution': ['Weibull'],
        'learning_rate': [1e-4],
        'discount': [0.5]
        # 'hidden_size': [[50], [50, 50], [100], [100, 100]],
        # 'k': [4, 6, 8],
        # 'distribution': ['LogNormal', 'Weibull'],
        # 'learning_rate': [1e-3, 1e-4],
        # 'discount': [0.5, 0.75, 1]
    })
})