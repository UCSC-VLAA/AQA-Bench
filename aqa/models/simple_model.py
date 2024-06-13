class SimpleModel():
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.reset()

    def reset(self):
        self.if_start = False
        self.last_guess = self.min - 1

    def __call__(self, prompt):
        if not self.if_start and "OK" in prompt:
            self.reset()
            return "OK"

        if prompt == "START":
            self.if_start = True

        if self.if_start:
            self.last_guess += 1
            return self.last_guess
