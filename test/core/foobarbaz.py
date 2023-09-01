class Foo:
    def __init__(self, value):
        self.value = value

    def hello(self):
        return self.value

    def add(self, add=1):
        self.value += add
