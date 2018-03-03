class Perceptron():

    def __init__(self):
        """Initialises a new classifier."""
        self.weights = {}
        self.acc = {}
        self.count = 1

    def predict(self, x, candidates=None):
        """Predicts the class, based on a feature vector

        Args:
            x: a feature vector
            candidates: possible classes to choose from

        Returns:
            The best predicted class, all classes with their respective scores
        """
        scores = {}
        for curr_class in self.weights:
            if not candidates or curr_class in candidates:
                scores[curr_class] = 0
                for tok in x:
                    if tok in self.weights[curr_class]:
                        scores[curr_class] += self.weights[curr_class][tok]
    
        best_score = -float("inf")
        best_class = "None"
        for curr_class,score in scores.items():
            if score >= best_score:
                best_score = score
                best_class = curr_class
        return best_class, scores
        

    def update(self, x, y):
        """Updates the weight vectors with a single training instance.

        Args:
            x: a feature vector
            y: The gold-standard class

        Returns:
            The predicted class
        """
        p, _ = self.predict(x)          
        if not p == y:
            if y not in self.weights:
                self.weights[y] = {}
                self.acc[y] = {}
            if p not in self.weights:
                self.weights[p] = {}
                self.acc[p] = {} 
            for key in x:
                if key not in self.weights[p]:
                    self.weights[p][key] = 0
                    self.acc[p][key] = 0
                self.weights[p][key] -= 1
                self.acc[p][key] -= self.count
                if key not in self.weights[y]:
                    self.weights[y][key] = 0
                    self.acc[y][key] = 0
                self.weights[y][key] += 1
                self.acc[y][key] += self.count
        self.count += 1
        return p 

    def finalize(self):
        """Finalizes the classifier by averaging its weight vectors."""
        for curr_class in self.weights:
            for tok in self.weights[curr_class]:
                self.weights[curr_class][tok] -= self.acc[curr_class][tok]/self.count
