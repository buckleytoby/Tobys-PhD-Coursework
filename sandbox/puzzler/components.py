from pygame.math import Vector2

from mixins import FrameMixin, IDMixin, ScreenMixin

from iomixins import IOMixin

from structs import Rect, Circle, Line, Point

import pygame

from constants import *

from screen import Screen

from io_ import PlayerInputs

from event_handlers import ClickAndDrag

import nodes

from transformations import rect_in_parent_frame, circle_in_parent_frame

# alias
class Component(IOMixin, FrameMixin, IDMixin, ScreenMixin):

    def draw(self):
        pass

### GEMINI GENERATED
class Slider(Component):
    """
    Slider
    """
    def __init__(self, 
                 xy: Vector2, # xy pos of the corner
                 wh: Vector2, # wh of the element aka the scale
                 label: str = "Value"
                 ) -> None:
        IOMixin.__init__(self)
        FrameMixin.__init__(self, xy, wh)
        IDMixin.__init__(self)
        ScreenMixin.__init__(self)

        # track
        self.track = Rect(SLATE_BLUE, 
                               Vector2(0.0, 0.4), # xy
                               Vector2(1.0, 0.2), # wh
                               )

        # handle
        self.handle = Circle(CORAL_DARK, 
                               Vector2(0.0, 0.5), # xy
                               radius = 0.1
                             )

        # parameters
        self.min_val = 0
        self.max_val = 1.0
        self.handle_radius = 0.1
        
        # State
        self.current_val = self.min_val
        self.dragging = False
        
        # Visuals
        self.label = label
        self.font = pygame.font.SysFont("Arial", 16)
        self.text_surf = self.font.render(f"{self.label}: {self.current_val:.2f}", True, (255, 255, 255))

        # setup my event handler
        self.event_handler = ClickAndDrag()

    def update(self):
        """
        
        """
        if self.dragging:
            mouse_x = self.io_node.get_mouse_pos()[0]

            # Constrain handle to track
            handle_x = max(0.0, min(mouse_x, 1.0))
            
            # Map handle position to value range
            self.handle.xy.x = handle_x

            # update state value
            self.current_val = handle_x

    def handle_event(self, event: pygame.event.Event):
        """
        passthrough
        """
        self.event_handler.handle_event(event, self.handle, self.handle_radius)

    def draw(self):
        # draw track
        self.screen_node.draw_rect(rect_in_parent_frame(self.track, self))

        # draw handle
        self.screen_node.draw_circle(circle_in_parent_frame(self.handle, self))

        # draw label
        self.screen_node.draw_text(self.text_surf)

    def get_value(self) -> float:
        return self.current_val
    
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
## end components
        