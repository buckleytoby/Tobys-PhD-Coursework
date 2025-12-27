from pygame.math import Vector2
import pygame

## structs
class Shape:
    id = 0
    def __init__(self) -> None:
        self.id = Shape.id
        Shape.id += 1

class Rect(Shape):
    """
    pygame.Rect assumes int parameters, so we must make our own with floating point values
    """
    def __init__(self,
                 color: tuple,
                 xy: Vector2,
                 wh: Vector2,
                 ) -> None:
        super().__init__()

        # save params
        self.color = color
        self.xy = xy
        self.wh = wh

    def copy(self):
        other = Rect(
            self.color,
            self.xy,
            self.wh
        )

        return other
    
    def get_pygame_rect(self):
        pygame_rect = pygame.Rect(
            self.xy,
            self.wh
        )

        return pygame_rect


class Line(Shape):
    def __init__(self,
                  color: tuple,
                  xy1: Vector2,
                  xy2: Vector2,
                  width: float = 0.1,
                 ) -> None:
        super().__init__()

        self.color = color
        self.xy1 = xy1
        self.xy2 = xy2
        self.width = width

class Circle(Shape):
    def __init__(self,
                 color: tuple,
                 xy: Vector2,
                 radius: float,
                 ) -> None:
        super().__init__()

        # save params
        self.color = color
        self.xy = xy
        self.radius = radius

    def copy(self):
        other = Circle(
            self.color,
            self.xy,
            self.radius
        )
        
        return other

# alias
class Point(Circle):
    def __init__(self, color: tuple, xy: Vector2, radius: float) -> None:
        super().__init__(color, xy, radius)
    

class ClassificationData:
    def __init__(self,
                 xys,
                 label,
                 ) -> None:
        self.xys = xys
        self.label = label

class State:
    """
    generic state struct
    """


    
def UnitRect(color):
    return Rect(
        color,
        xy = Vector2(0, 0),
        wh = Vector2(1.0, 1.0),
    )

def UnitXLine(color):
    return Line(
        color,
        Vector2(0, 0.5),
        Vector2(1.0, 0.5),
    )