import pygame
from pygame.math import Vector2

from structs import Rect, Line, Circle, Point
from utils import vector2_to_inttuple

from sprites import Sprite

import nodes

class Screen:
    def __init__(self) -> None:
        
        # my members
        self.pygame_screen = None
        self.screen_pixel_wh = Vector2(960, 720) # pixels
        self.sprites = pygame.sprite.Group()

        # self.scale = Vector2(640.0, 480.0) # TODO: switch to scale
        self.zoom_level = Vector2(320.0, 240.0)  # screen viewport's zoom level

        self.viewport_xy = Vector2(0.0, 0.0) # the screen viewport's coordinates

        # init myself
        self.init()

    def get_screen_wh(self):
        wh = self.pygamewh_to_wh(self.screen_pixel_wh)

        return wh

    def wh_to_pygamewh(self, wh: Vector2):
        # straight multiplication
        pygamewh = wh.elementwise() * (self.zoom_level.elementwise())
        
        return pygamewh
    
    def xy_to_pygamexy(self, xy: Vector2, pygameh):
        # multiplication and offset y = ax + b
        pygamexy = self.wh_to_pygamewh(xy)

        # TODO: add the offset

        # y val must be flipped, and must subtract off height
        pygamexy.y = self.screen_pixel_wh.y - pygamexy.y - pygameh

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
        pygamexy.y = self.screen_pixel_wh.y - pygamexy.y - pygameh

        # scale the xy
        xy = self.pygamewh_to_wh(pygamexy)

        return xy
        

    def init(self):
        self.pygame_screen = pygame.display.set_mode(vector2_to_inttuple(self.screen_pixel_wh))

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

    def draw_text(self, text_surface, xy: Vector2):
        assert(self.pygame_screen is not None)

        xy2 = self.xy_to_pygamexy(xy, 0)

        self.pygame_screen.blit(text_surface, xy2)

    def add_sprite(self, sprite):
        self.sprites.add(sprite)

    def draw_sprite(self, sprite: Sprite):
        assert(self.pygame_screen is not None)

        pixel_wh = self.wh_to_pygamewh(sprite.wh)

        xy2 = self.xy_to_pygamexy(sprite.xy, pixel_wh[1])

        # mutable copy
        surf = sprite.image.copy()

        # scale
        surf = pygame.transform.scale(surf, pixel_wh)


        self.pygame_screen.blit(surf, xy2)

    # def update_sprites(self):
    #     for sprite in self.sprites:
    #         sprite.get_pygame_rect().xy = 

    # def draw_sprites(self):
    #     assert(self.pygame_screen is not None)

    #     # must update internal pygame rect
    #     self.update_sprites()

    #     # convenience draw function
    #     self.sprites.draw(self.pygame_screen)


    
class ScreenMixin:
    def __init__(self) -> None:
        
        # node refs
        assert(isinstance(nodes.SCREEN_NODE, Screen))
        self.screen_node: Screen = nodes.SCREEN_NODE