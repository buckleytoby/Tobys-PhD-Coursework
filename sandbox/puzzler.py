import pygame

from pygame.math import Vector2

# globals
game = None
# end globals

# some constants
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
# end some constants

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


class Line(Shape):
    def __init__(self,
                  color: tuple,
                  xy1: pygame.math.Vector2,
                  xy2: pygame.math.Vector2,
                  width: float = 1,
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
    
## end structs

## utils
def vector2_to_inttuple(xy: Vector2):
    x = int(xy.x)
    y = int(xy.y)
    return x, y
## end utils

## parent classes
class Base:
    def run(self):
        pass

class FrameMixin:
    def __init__(self,
                 xy: Vector2,
                 wh: Vector2,
                 ) -> None:
        self.xy = xy # my xy relative to my parent
        self.wh = wh # my scale relative to my parent's scale


    def scale_to_parentscale(self, xy: Vector2):
        # straight multiplication
        parentxy = xy.elementwise() * (self.wh.elementwise())
        
        return parentxy
    
    def xy_to_parentxy(self, xy: Vector2):
        """
        transform xy in this frame to xy w.r.t. my parent
        """
        # multiplication and offset y = ax + b
        parentxy = self.scale_to_parentscale(xy) + self.xy

        return parentxy

## end parent classes

class PlayerInputs:
    def __init__(self) -> None:
        
        
        self.mapping = {}

        # load default mapping
        self.default_mapping()

    def default_mapping(self):
        self.mapping = {
            "right": pygame.K_RIGHT,
        }

    def get_pygame_key(self, map_key):
        return self.mapping[map_key]

    def is_pressed(self, map_key):
        pygame_key = self.get_pygame_key(map_key)

    def is_pygame_pressed(self, pygame_key):
        keys = pygame.key.get_pressed()

        if pygame_key in keys:
            return True
        else:
            return False
        
    def process_some_events(self):
        pass
        
class Screen:
    def __init__(self) -> None:
        
        # my members
        self.pygame_screen = None
        self.screen_wh = Vector2(640, 480) # pixels

        # self.scale = Vector2(640.0, 480.0) # TODO: switch to scale
        self.zoom_level = Vector2(640.0, 480.0)  # screen viewport's zoom level

        self.viewport_xy = Vector2(0.0, 0.0) # the screen viewport's coordinates

        # init myself
        self.init()

    def wh_to_pygamewh(self, wh: Vector2):
        # straight multiplication
        pygamewh = wh.elementwise() * (self.zoom_level.elementwise())
        
        return pygamewh
    
    def xy_to_pygamexy(self, xy: Vector2, pygameh):
        # multiplication and offset y = ax + b
        pygamexy = self.wh_to_pygamewh(xy)

        # y val must be flipped, and must subtract off height
        pygamexy.y = self.screen_wh.y - pygamexy.y - pygameh

        return pygamexy
    
    def rect_to_pygamerect(self, rect: Rect):
        pygamewh = self.wh_to_pygamewh(rect.wh)
        pygamexy = self.xy_to_pygamexy(rect.xy, pygamewh.y)

        r = pygame.Rect(
            pygamexy,
            pygamewh
        )
        return r
        

    def init(self):
        self.pygame_screen = pygame.display.set_mode(vector2_to_inttuple(self.screen_wh))

        pygame.display.set_caption("Toby's Puzzling Puzzler")

    def reset(self):
        assert(self.pygame_screen is not None)
        # reset screen
        self.pygame_screen.fill((150, 150, 150)) # light gray

    def display(self):
        pygame.display.flip()

    def draw_rect(self, 
                  rect: Rect
                  ):
        assert(self.pygame_screen is not None)

        # convert to pygame
        pygame_rect = self.rect_to_pygamerect(rect)

        # draw
        pygame.draw.rect(self.pygame_screen, rect.color, pygame_rect)

    def draw_line(self,
                  line: Line
                  ):
        assert(self.pygame_screen is not None)

        # convert to pygame
        wh = Vector2(line.width, line.width)
        pygame_wh = self.wh_to_pygamewh(wh)

        xy1 = self.xy_to_pygamexy(line.xy1, pygame_wh.y)
        xy2 = self.xy_to_pygamexy(line.xy2, pygame_wh.y)

        # draw
        pygame.draw.line(self.pygame_screen, line.color, xy1, xy2, int(pygame_wh.x))

    def draw_circle(self,
                    circle: Circle,
                ):
        assert(self.pygame_screen is not None)

        # convert to pygame
        rr = Vector2(circle.radius, circle.radius)

        xy = self.xy_to_pygamexy(circle.xy, 0.0)
        rr2 = self.wh_to_pygamewh(rr)

        # draw
        pygame.draw.circle(self.pygame_screen, circle.color, xy, rr2.x)

    # alias
    def draw_point(self, point: Point):
        self.draw_circle(point)
        

    def draw(self):
        self.reset()

class LevelMap:
    def __init__(self) -> None:
        
        # references
        assert(game is not None)
        self.screen = game.screen

        # my members
        self.reset()

    def reset(self):
        self.rects = {}
        self.surfaces = {}
        self.lines = {}
        self.points = {}

    def draw_rects(self):
        rect: Rect
        for id, rect in self.rects.items():
            self.screen.draw_rect(rect)

    def draw_lines(self):
        line: Line
        for id, line in self.lines.items():
            self.screen.draw_line(line)

    def draw_points(self):
        point: Point
        for id, point in self.points.items():
            self.screen.draw_point(point)


    def draw(self):
        # rects
        self.draw_rects()

        # lines
        self.draw_lines()

        # circles
        self.draw_points()

    def add_rect(self, rect: Rect):
        self.rects[rect.id] = rect

    def add_line(self, line: Line):
        self.lines[line.id] = line

    def add_point(self, point: Point):
        self.points[point.id] = point

class Chart(FrameMixin):
    def __init__(self,
                 xy: Vector2,
                 wh: Vector2
                 ) -> None:
        FrameMixin.__init__(self, xy, wh)

        # my members
        self.points: list[Point] = list()

    def add_point(self, xy: Point):
        self.points.append(xy)

    def add_points(self, xys: list):
        for xy in xys:
            self.add_point(xy)

    def get_x_axis_line(self):
        xy1 = Vector2(-0.01, 0.0) # xy1
        xy2 = Vector2(1.0, 0.0) # xy2

        line = Line(
            GREEN,
            xy1,
            xy2,
            width = 0.05,
        )

        return line

    def get_y_axis_line(self):
        xy1 = Vector2(0.0, -0.01) # xy1
        xy2 = Vector2(0.0, 1.0) # xy2

        line = Line(
            GREEN,
            xy1,
            xy2,
            width = 0.05,
        )

        return line

    def get_parent_line(self, line: Line):
        """
        
        """
        line.xy1 = self.xy_to_parentxy(line.xy1)
        line.xy2 = self.xy_to_parentxy(line.xy2)

        return line
    
    def get_parent_x_axis(self):
        line = self.get_x_axis_line()

        line = self.get_parent_line(line)

        return line
    
    def get_parent_y_axis(self):
        line = self.get_y_axis_line()

        line = self.get_parent_line(line)

        return line
    
    def get_parent_pts(self):
        ppts = []

        for pt in self.points:
            parent_xy = self.xy_to_parentxy(pt.xy)

            ppt = pt.copy()
            ppt.xy = parent_xy

            ppts.append(ppt)

        return ppts

        

class Level:
    def draw(self):
        pass

class Level1(Level):
    """
    Classification 1
    """
    def __init__(self) -> None:
        super().__init__()

        # references

        # my members
        self.map = LevelMap()

        self.init()

    def init(self):
        # set up this level's unique map
        self.map.reset()

        # lvl1 xys
        xys = [
            Point(RED, Vector2(0.5, 0.25), radius = 0.01), 
            Point(BLUE, Vector2(0.5, 0.75), radius = 0.01), 
        ]

        # chart location
        chart = Chart(
            Vector2(0.1, 0.1), # location
            Vector2(1.0, 1.0), # scale
        )

        # 
        chart.add_points(xys)

        # add chart to map
        self.map.add_line(chart.get_parent_x_axis())
        self.map.add_line(chart.get_parent_y_axis())

        for pt in chart.get_parent_pts():
            self.map.add_point(pt)



    def draw(self):
        self.map.draw()



class Game(Base):
    """
    Top-level 7 things
    """
    def __init__(self,
                 dt = 1/60,
                 fps = 60,
                 ) -> None:
        # my parameters
        self.dt = dt
        self.fps = fps
        
        # my <=7 members
        self.inputs = PlayerInputs()
        self.screen = Screen()
        self.clock = pygame.time.Clock()

    def draw(self):
        # draw the screen
        self.screen.draw()

        # draw the level
        if isinstance(self.active_level, Level):
            self.active_level.draw()



    def step(self):
        # process some events
        self.inputs.process_some_events()

        # draw
        self.draw()

        # render
        self.screen.display()

        # tick
        self.clock.tick(self.fps)


    def run(self):
        """
        infinite game loop
        """
        done = False
        while not done:
            self.step()

            pass

    def load_level(self, level_class):
        # make an instance
        lvl = level_class()

        # save it
        self.active_level = lvl


def main():
    global game

    # spool up global level class instances
    pygame.init()

    game = Game()

    # load a level
    game.load_level(Level1)



    # run the game loop
    game.run()




if __name__ == "__main__":
    main()