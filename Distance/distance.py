from math import sqrt


class SqrtDistance:

    def distance(self, vec1, vec2):
        vec = vec1 - vec2
        total = 0.0
        for element in vec:
            total += pow(element, 2)
        return sqrt(total)

