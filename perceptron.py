class Perceptron():

    def __init__(self):
        """Initialises a new classifier."""
        self.weights = {}
        self.acc = {}
        self.count = 1

    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
        scores = {}
        for curr_class in self.weights:
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
        return best_class
        

    def update(self, x, y):
        """Updates the weight vectors with a single training instance.

        Args:
            x: A document, represented as a list of words.
            y: The gold-standard class, represented as a string.

        Returns:
            The predicted class, represented as a string.
        """
        p = self.predict(x)          
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
        for curr_class in self.weights:
            for tok in self.weights[curr_class]:
                self.weights[curr_class][tok] -= self.acc[curr_class][tok]/self.count
