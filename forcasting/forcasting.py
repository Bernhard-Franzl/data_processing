
class Forcaster():
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def forecast(self, steps):
        return self.model.predict(steps)
    
class Clustering():
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def cluster(self):
        return self.model.fit(self.data)