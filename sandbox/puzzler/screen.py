import pygame
from pygame.math import Vector2
from structs import Rect, Line, Circle, Point
from utils import vector2_to_inttuple

class Screen:
    def __init__(self) -> None:
        
        # my members
        self.pygame_screen = None
        self.screen_wh = Vector2(640, 480) # pixels

        # self.scale = Vector2(640.0, 480.0) # TODO: switch to scale
        self.zoom_level = Vector2(320.0, 240.0)  # screen viewport's zoom level

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

        # TODO: add the offset

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
    
    def pygamewh_to_wh(self, pygamewh: Vector2):
        """
        inverse of wh_to_pygamewh
        """
        wh = pygamewh.elementwise() / (self.zoom_level.elementwise())
        
        return wh

    
    def pygamexy_to_xy(self, pygamexy: Vector2, pygameh):
        """
        inverse of xy_to_pygamexy
        """
        # mutable copy
        pygamexy = pygamexy.copy()

        # y val must be flipped, and must subtract off height
        pygamexy.y = self.screen_wh.y - pygamexy.y - pygameh

        # scale the xy
        xy = self.pygamewh_to_wh(pygamexy)

        return xy
        

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

    def draw_text(self, text_surface):
        assert(self.pygame_screen is not None)
        self.pygame_screen.blit(text_surface, (50, 50))
