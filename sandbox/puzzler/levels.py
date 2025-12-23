from mixins import IDMixin, ScreenMixin

from iomixins import EventHandlerMixin

from structs import Rect, Line, Point, Circle

from pygame.math import Vector2

from constants import *

import nodes

import pygame

from components import Component, Slider, Chart

from utils import get_random_rgb
     
class LevelMap(ScreenMixin):
    """
    contains the graphics for the level
    
    TODO: move shapes to components, just do components here (or whatever)
    """
    def __init__(self) -> None:
        ScreenMixin.__init__(self)

        # my members
        self.reset()

    def reset(self):
        self.rects = {}
        self.surfaces = {}
        self.lines = {}
        self.points = {}
        self.texts = []
        self.components = {}

    def draw_rects(self):
        rect: Rect
        for id, rect in self.rects.items():
            self.screen_node.draw_rect(rect)

    def draw_lines(self):
        line: Line
        for id, line in self.lines.items():
            self.screen_node.draw_line(line)

    def draw_points(self):
        point: Point
        for id, point in self.points.items():
            self.screen_node.draw_point(point)

    def draw_texts(self):
        for text in self.texts:
            self.screen_node.draw_text(text)

    def draw_components(self):
        component: Component
        for id, component in self.components.items():
            component.draw()


    def draw(self):
        # rects
        self.draw_rects()

        # lines
        self.draw_lines()

        # circles
        self.draw_points()

        self.draw_texts()

        self.draw_components()

    def add_rect(self, rect: Rect):
        self.rects[rect.id] = rect

    def add_line(self, line: Line):
        self.lines[line.id] = line

    def add_point(self, point: Point):
        self.points[point.id] = point

    def add_text(self, text):
        self.texts.append(text)

    def add_component(self, component: Component):
        self.components[component.id] = component

    def update(self):
        # update all my components
        for id, component in self.components.items():
            component.update()



class Connector(IDMixin):
    def __init__(self,
                 upstream_fcn,
                 downstream_fcn,
                 ):
        IDMixin.__init__(self)

        self.upstream_fcn = upstream_fcn
        self.downstream_fcn = downstream_fcn

    def update(self):
        outputs = self.upstream_fcn()

        self.downstream_fcn(outputs)

class LinearAffine:
    def __init__(self,
                 a,
                 b,
                 ) -> None:
        self.a = a
        self.b = b

    def forward(self, x):
        y = self.a * x + self.b
        return y
    
class Linear(LinearAffine):
    def __init__(self, a) -> None:
        super().__init__(a, b = 0.0)



class Level(EventHandlerMixin):
    def __init__(self) -> None:
        
        
        # my members
        self.map = None
        self.updateable_children = {}

    def draw(self):
        pass

    def update_children(self):
        for id, child in self.updateable_children.items():
            child.update()

    def update(self):
        assert(self.map is not None)

        # update my map
        self.map.update()

        # update my updateable children
        self.update_children()

    def add_updateable_child(self, child):
        self.updateable_children[child.id] = child




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





        # level instructions
        my_font = pygame.font.SysFont("Arial", 20)
        text_surface = my_font.render("Level 1: Classification. Move the slider to separate the data", True, (255, 255, 255))
        self.map.add_text(text_surface)

        # control slider
        slider = Slider(
            xy = Vector2(1.0, 0.5),
            wh = Vector2(1.0, 1.0),
            label = "Bias Value."
        )
        self.slider = slider

        # add for drawing
        self.map.add_component(slider)

        # response equation
        equation = LinearAffine(0.0, 0.0)

        # separator line
        line = Line(get_random_rgb(), Vector2(0.0, 0.0), Vector2(1.0, 0.0))

        # add for drawing
        self.map.add_line(line)

        # connector1 fcn
        def fcn(b):
            equation.b = b

        # connector2 fcns
        def get_b():
            return equation.b
        
        def update_line(b):
            # move horizontally w.r.t. bias, b
            line.xy1.y = b
            line.xy2.y = b

        # connector object, slider value to line
        connector1 = Connector(slider.get_value, fcn)
        connector2 = Connector(get_b, update_line)

        self.add_updateable_child(connector1)
        self.add_updateable_child(connector2)




    def draw(self):
        self.map.draw()

    def handle_event(self, event):
        """
        
        """
        self.slider.handle_event(event)

