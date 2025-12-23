import random
from pygame.math import Vector2

## utils
def vector2_to_inttuple(xy: Vector2):
    x = int(xy.x)
    y = int(xy.y)
    return x, y

# GEMINI GENERATED
def get_random_rgb():
    """Returns a tuple of three random integers between 0 and 255."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)
## end utils