from easydict import EasyDict



########################## Deep Survival Machines ##########################

### Metabric ###
DSM_metabric = EasyDict({
    'data': 'metabric',
    'horizons': [.25, .5, .75],
    'batch_size': 64,
    'epochs': 1,
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
    'epochs': 1,
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
    'epochs': 1,
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