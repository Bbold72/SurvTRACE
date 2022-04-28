from sksurv.ensemble import RandomSurvivalForest

class RSF:

    def __init__(self, config):

        self.model = RandomSurvivalForest(n_estimators=config.epochs, 
                                            verbose=1,
                                            max_depth=4,
                                            n_jobs=-1
                                            )

    def train(self, data):
        return self.model.fit(data.x_train, data.y_et_train)