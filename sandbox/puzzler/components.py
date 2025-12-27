import numpy as np

from pygame.math import Vector2

from mixins import FrameMixin, IDMixin, ScreenMixin

from iomixins import IOMixin

from structs import Rect, Circle, Line, Point, UnitRect, UnitXLine

import pygame

from constants import *

from screen import Screen

from io_ import PlayerInputs

from event_handlers import Click, ClickAndDrag

import nodes

from transformations import rect_in_parent_frame, circle_in_parent_frame

from utils import get_random_rgb

# alias
class Component(IOMixin, FrameMixin, IDMixin, ScreenMixin):
    def __init__(self,
                 xy = Vector2(0, 0), # xy pos of the corner
                 wh = Vector2(1, 1), # wh of the element aka the scale
                 use_background = False
                 ) -> None:
        IOMixin.__init__(self)
        FrameMixin.__init__(self, xy, wh)
        IDMixin.__init__(self)
        ScreenMixin.__init__(self)

        self.use_background = use_background or DEBUG_MODE
        self.bg_color = get_random_rgb()

        self.drawables = []
        self.updateables = []

    def add_drawable(self, d):
        self.drawables.append(d)

    def draw(self):
        # background
        r = UnitRect(self.bg_color)
        self.screen_node.draw_rect(rect_in_parent_frame(r, self))

        # drawables
        for drawable in self.drawables:
            drawable.draw()

    def update(self):
        pass

class LineComponent(Component):
    def __init__(self, xy=Vector2(0, 0), wh=Vector2(1, 1), use_background=False,
                 color = get_random_rgb(),
                 ) -> None:
        super().__init__(xy, wh, use_background)

        self.line = UnitXLine(color)

    def draw(self):
        self.screen_node.draw_line(self.line)

class LineComponentPassIn(Component):
    def __init__(self, xy=Vector2(0, 0), wh=Vector2(1, 1), use_background=False,
                 line = UnitXLine(get_random_rgb()),
                 ) -> None:
        super().__init__(xy, wh, use_background)

        self.line = line

    def draw(self):
        self.screen_node.draw_line(self.line)

### GEMINI GENERATED, toby modified
class Slider(Component):
    """
    Slider
    """
    def __init__(self, 
                 xy: Vector2, # xy pos of the corner
                 wh: Vector2, # wh of the element aka the scale
                 label: str = "Value"
                 ) -> None:
        Component.__init__(self, xy, wh)

        self.label = label

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
        self.io_handler = ClickAndDrag()

    def update(self):
        """
        
        """
        if self.dragging:
            # global mouse_x
            mouse_globalxy = self.io_node.get_mouse_pos()

            mouse_xy = self.parentxy_to_xy(mouse_globalxy)

            mouse_x = mouse_xy[0]

            # Constrain handle to track
            handle_x = max(0.0, min(mouse_x, 1.0))
            
            # Map handle position to value range
            self.handle.xy.x = handle_x

            # update state value
            self.current_val = handle_x

            # update text surf
            self.text_surf = self.font.render(f"{self.label}: {self.current_val:.2f}", True, (255, 255, 255))

        

    def handle_event(self, event: pygame.event.Event):
        """
        passthrough
        """
        # get handle global xy
        parentxy = self.xy_to_parentxy(self.handle.xy)

        self.io_handler.handle_event(event, self, parentxy, self.handle_radius)

    def draw(self):
        Component.draw(self)

        # draw track
        self.screen_node.draw_rect(rect_in_parent_frame(self.track, self))

        # draw handle
        self.screen_node.draw_circle(circle_in_parent_frame(self.handle, self))

        # draw label
        self.screen_node.draw_text(self.text_surf, self.xy_to_parentxy(Vector2(0, 0)))

    def get_value(self) -> float:
        return self.current_val
    

class Axis(Component):
    def __init__(self, xy: Vector2, wh: Vector2, use_background=False,
                 nb_ticks = 1,
                 ) -> None:
        super().__init__(xy, wh, use_background)

        self.nb_ticks = nb_ticks

        w, h = self.screen_node.get_screen_wh()

        # axis
        line = Line(
            GREEN,
            xy1 = Vector2(0, 0.05),
            xy2 = Vector2(w, 0.05),
            width = 0.05,
        )
        self.add_drawable(LineComponentPassIn(line = line))

        # ticks
        tickxs = np.linspace(0, w, nb_ticks)

        for tickx in tickxs:
            line = Line(
                GREEN,
                Vector2(tickx, 0.0),
                Vector2(tickx, 0.2),
                width = 0.025
            )
            self.add_drawable(LineComponentPassIn(line = line))

class Grid(Axis):
    def __init__(self, nb_ticks=1) -> None:
        super().__init__(
            xy = Vector2(0, 0.0), # translate
            wh = Vector2(1, 1), # unit
            nb_ticks = nb_ticks
        )

class Clicker(Component):
    def __init__(self, 
                 xy: Vector2, # xy pos of the corner
                 wh: Vector2, # wh of the element aka the scale
                 clicked_cb,
                 label: str = "Value",
                 ) -> None:
        Component.__init__(self, xy, wh)
        
        # my members
        self.clicked_cb = clicked_cb

        # the interactable part
        self.interactable = Rect(SLATE_BLUE, 
                               Vector2(-0.5, -0.5), # xy
                               Vector2(1.0, 1.0), # wh
                               )
        
        # State
        self.clicked = False
        
        # Visuals
        self.label = label
        self.font = pygame.font.SysFont("Arial", 16)
        self.text_surf = self.font.render(f"{self.label}: {self.clicked:.2f}", True, (255, 255, 255))

        # setup my io handler
        self.io_handler = Click()

    # no update needed
    # def update(self):
        # pass

    def handle_event(self, event: pygame.event.Event):
        # get interactable global xy
        rect_global = rect_in_parent_frame(self.interactable, self)

        # I/O
        self.io_handler.handle_event(event, self, rect_global)

        # if clicked
        if self.clicked:
            if self.clicked_cb is not None:
                self.clicked_cb()

    def draw(self):
        Component.draw(self)

        # draw interactable
        self.screen_node.draw_rect(rect_in_parent_frame(self.interactable, self))

        # draw label
        self.screen_node.draw_text(self.text_surf, self.xy_to_parentxy(Vector2(-0.5, 0.0)))

    def get_value(self):
        return self.clicked
    
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
        