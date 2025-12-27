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

# GEMINI GENERATED
def get_signed_distance(point: Vector2, 
                        line_start: Vector2, 
                        line_end: Vector2):
    A = Vector2(line_start)
    B = Vector2(line_end)
    P = Vector2(point)
    
    # 1. Get the vector of the line itself
    line_vec = B - A
    
    if line_vec.length() == 0:
        return P.distance_to(A)
    
    # 2. Get the normal vector (perpendicular to the line)
    # rotate(90) gives a vector pointing to one side
    # We normalize it so the result is in actual pixels/units
    normal = Vector2(-line_vec.y, line_vec.x).normalize()
    
    # 3. Vector from line start to the point
    point_vec = P - A
    
    # 4. The Dot Product of the point_vec and the normal
    # This returns the length of point_vec projected onto the normal
    signed_dist = point_vec.dot(normal)
    
    return signed_dist