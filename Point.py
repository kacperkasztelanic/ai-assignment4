class Point:
    def __init__(self, coords, vector, id):
        self.coords = coords
        self.vector = vector
        self.id = id

    def __str__(self):
        return str(self.id) + "|" + str(self.coords)

    def __repr__(self):
        return str(self.id) + "|" + str(self.coords)
