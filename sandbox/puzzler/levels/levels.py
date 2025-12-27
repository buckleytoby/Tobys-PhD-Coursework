import numpy as np

from mixins import IDMixin, ScreenMixin

from iomixins import EventHandlerMixin, IOMixin

from structs import Rect, Line, Point, Circle, ClassificationData, State

from pygame.math import Vector2

from constants import *

import nodes

import pygame

from components import Component, Slider, Chart, Clicker, Grid

from utils import get_random_rgb, get_signed_distance

     
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
            self.screen_node.draw_text(text, Vector2(0.1, 1.9))

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

class Equation:
    def forward(self, x):
        return 0.0
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class LinearAffine(Equation):
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

class LeakyReLU(Equation):
    def __init__(self,
                 neg_scale = 0.1,
                 ) -> None:
        super().__init__()

        self.neg_scale = neg_scale

    def forward(self, x):
        if x >= 0.0:
            return x
        else:
            return self.neg_scale * x


class PiecewiseLinear(Equation):
    """
    simplest non-linear function
    y1 = ax + b
    y2 = leakyReLu(y1)
    y3 = ay2 + b
    """
    def __init__(self) -> None:
        super().__init__()

        self.linear1 = LinearAffine(0, 0)
        self.relu = LeakyReLU()
        self.linear2 = LinearAffine(0, 0)

    def forward(self, x):
        y1 = self.linear1(x)
        y2 = self.relu(y1)
        y3 = self.linear2(y2)

        return y3

class LevelSuccess(IOMixin):
    def update(self):
        pass

class TestLevelSuccess1(LevelSuccess):
    def check(self):
        return self.io_node.is_pygame_pressed(pygame.K_SPACE)

class LevelSuccess1(LevelSuccess):
    def get_dists_list(self, xy1, xy2, data: ClassificationData):
        dists = []
        pt: Point
        for pt in data.xys:
            dist = get_signed_distance(pt.xy, xy1, xy2)

            dists.append(dist)

        return dists

    def check(self, equation: Equation, data1: ClassificationData, data2: ClassificationData):

        xy1 = Vector2(0.0, equation.forward(0.0))
        xy2 = Vector2(1.0, equation.forward(1.0))

        dists1 = self.get_dists_list(xy1, xy2, data1)
        dists2 = self.get_dists_list(xy1, xy2, data2)

        is_same_sign1 = len(np.unique(np.sign(dists1))) == 1
        is_same_sign2 = len(np.unique(np.sign(dists2))) == 1

        diff_signs = np.sign(dists1[0]) != np.sign(dists2[0])

        if is_same_sign1 and is_same_sign2 and diff_signs:
            return True
        else:
            return False

class Level(EventHandlerMixin):
    def __init__(self) -> None:
        super().__init__()
        
        # my members
        self.map = LevelMap()
        self.updateable_children = {}
        self.level_success = LevelSuccess()
        self.is_completed = False

        self.init()

    def init(self):
        pass

    def draw(self):
        self.map.draw()

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


class ClassificationLevel(Level):
    """
    Classification
    """

    def init(self):
        # set up this level's unique map
        self.map.reset()

        # lvl xys
        xys1 = [
            Point(RED, Vector2(0.5, 0.25), radius = 0.01), 
        ]
        xys2 = [
            Point(BLUE, Vector2(0.5, 0.75), radius = 0.01), 
        ]
        self.data1 = ClassificationData(xys1, label = "red")
        self.data2 = ClassificationData(xys2, label = "blue")

        xys = xys1 + xys2

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


        # control slider
        slider = Slider(
            xy = Vector2(1.0, 0.5),
            wh = Vector2(1.0, 1.0),
            label = "Scale."
        )
        self.slider = slider

        # add for drawing
        self.map.add_component(slider)

        # response equation
        equation = LinearAffine(0.0, 0.0)
        self.equation = equation


        self.level_success = LevelSuccess1()


        def clicked_cb():
            print("clicked.")
            self.save_if_completed()
                
        # clicker
        clicker = Clicker(
            xy = Vector2(1.5, 1.5),
            wh = Vector2(0.1, 0.1),
            clicked_cb = clicked_cb,
            label = "Check Solution.",
        )
        self.clicker = clicker

        self.map.add_component(clicker)

    def draw(self):
        self.map.draw()

    def handle_event(self, event):
        """
        """
        self.slider.handle_event(event)
        self.clicker.handle_event(event)

    def save_if_completed(self): #type:ignore
        self.is_completed = self.level_success.check(self.equation, self.data1, self.data2)



class TestLevel1(Level):
    """
    test only
    """
    def init(self):


        # level instructions
        my_font = pygame.font.SysFont("Arial", 20)
        text_surface = my_font.render("TEST LEVEL 1.", True, (255, 255, 255))
        self.map.add_text(text_surface)

        # grid
        grid = Grid(10)
        self.map.add_component(grid)

        ## basic shapes
        # rect = Rect(get_random_rgb(), )

        # separator line
        line = Line(SILVER, xy1 = Vector2(0.1, 0.1), xy2 = Vector2(1.1, 0.1), width = 0.05)

        # add for drawing
        self.map.add_line(line)


        self.level_success = TestLevelSuccess1()

        def f():
            # if space pressed
            pressed = self.io_node.is_pygame_pressed(pygame.K_SPACE)

            # save
            if pressed:
                self.is_completed = pressed

        class UpdateCallsCB(IDMixin):
            def __init__(self,
                         f,
                         ) -> None:
                IDMixin.__init__(self)

                self.f = f

            def update(self):
                self.f()

        c = UpdateCallsCB(f)

        self.add_updateable_child(c)


class Level1(ClassificationLevel):
    """
    bias only
    """
    def init(self):
        ClassificationLevel.init(self)

        self.slider.label = "Bias"

        # level instructions
        my_font = pygame.font.SysFont("Arial", 20)
        text_surface = my_font.render("Level 1: Classification. Move the slider to separate the data", True, (255, 255, 255))
        self.map.add_text(text_surface)

        # separator line
        line = Line(SILVER, xy1 = Vector2(0.1, 0.1), xy2 = Vector2(1.1, 0.1), width = 0.05)

        # add for drawing
        self.map.add_line(line)

        # connector1 fcn
        def fcn(b):
            self.equation.b = b

        # connector2 fcns
        def get_b():
            return self.equation.b
        
        def update_line(b):
            # move horizontally w.r.t. bias, b
            line.xy1.y = b + 0.1
            line.xy2.y = b + 0.1 # hard coded offset for alignment

        # connector object, slider value to line
        connector1 = Connector(self.slider.get_value, fcn)
        connector2 = Connector(get_b, update_line)

        self.add_updateable_child(connector1)
        self.add_updateable_child(connector2)


class Level2(ClassificationLevel):
    """
    scale only
    """
    def init(self):
        ClassificationLevel.init(self)

        # level instructions
        my_font = pygame.font.SysFont("Arial", 20)
        text_surface = my_font.render("Level 2: Classification. Move the slider to separate the data", True, (255, 255, 255))
        self.map.add_text(text_surface)

        # separator line
        line = Line(SILVER, xy1 = Vector2(0.1, 0.1), xy2 = Vector2(1.1, 0.1), width = 0.05)

        # add for drawing
        self.map.add_line(line)

        # connector1 fcn
        def fcn(a):
            self.equation.a = a

        # connector2 fcns
        def get_a():
            return self.equation.a
        
        def update_line(a):
            # update xy1 and xy2
            line.xy1.y = self.equation.forward(line.xy1.x)
            line.xy2.y = self.equation.forward(line.xy2.x)

        # connector object, slider value to line
        connector1 = Connector(self.slider.get_value, fcn)
        connector2 = Connector(get_a, update_line)

        self.add_updateable_child(connector1)
        self.add_updateable_child(connector2)




class Level3(ClassificationLevel):
    """
    scale and bias
    """
    def init(self):
        # set up this level's unique map
        self.map.reset()

        """ Data """
        xys1 = [
            Point(RED, Vector2(0.5, 0.25), radius = 0.01), 
            Point(RED, Vector2(0.6, 0.2), radius = 0.01), 
        ]
        xys2 = [
            Point(BLUE, Vector2(0.4, 0.6), radius = 0.01), 
            Point(BLUE, Vector2(0.5, 0.75), radius = 0.01), 
            Point(BLUE, Vector2(0.7, 0.8), radius = 0.01), 
            Point(BLUE, Vector2(0.9, 0.4), radius = 0.01), 
        ]
        self.data1 = ClassificationData(xys1, label = "red")
        self.data2 = ClassificationData(xys2, label = "blue")


        xys = xys1 + xys2

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


        # control slider
        slider = Slider(
            xy = Vector2(1.0, 0.5),
            wh = Vector2(1.0, 1.0),
            label = "Scale."
        )
        self.slider = slider

        # add for drawing
        self.map.add_component(slider)

        # response equation
        equation = LinearAffine(0.0, 0.0)
        self.equation = equation


        self.level_success = LevelSuccess1()


        def clicked_cb():
            print("clicked.")
            self.save_if_completed()
                
        # clicker
        clicker = Clicker(
            xy = Vector2(1.5, 1.5),
            wh = Vector2(0.1, 0.1),
            clicked_cb = clicked_cb,
            label = "Check Solution.",
        )
        self.clicker = clicker

        self.map.add_component(clicker)


        
        """ level instructions """
        my_font = pygame.font.SysFont("Arial", 20)
        text_surface = my_font.render("Level 3: Classification. Move the slider to separate the data", True, (255, 255, 255))
        self.map.add_text(text_surface)

        # separator line
        line = Line(SILVER, xy1 = Vector2(0.1, 0.1), xy2 = Vector2(1.1, 0.1), width = 0.05)

        # add for drawing
        self.map.add_line(line)

        ###
        # add for drawing
        self.map.add_component(self.slider)

        # connector1 fcn
        def fcn(a):
            self.equation.a = a * 2.0 - 1.0

        # connector2 fcns
        def get_a():
            return self.equation.a
        
        def update_line(_):
            # update xy1 and xy2
            line.xy1.y = self.equation.forward(line.xy1.x)
            line.xy2.y = self.equation.forward(line.xy2.x)

        # connector object, slider value to line
        connector1 = Connector(self.slider.get_value, fcn)
        connector2 = Connector(get_a, update_line)

        self.add_updateable_child(connector1)
        self.add_updateable_child(connector2)


        ###
        # control slider
        slider2 = Slider(
            xy = Vector2(1.0, 0.25),
            wh = Vector2(1.0, 1.0),
            label = "Bias Value."
        )
        self.slider2 = slider2

        # add for drawing
        self.map.add_component(slider2)

        # connector1 fcn
        def fcn3(b):
            self.equation.b = b * 2.0 - 1.0

        # connector2 fcns
        def get_b():
            return self.equation.b

        # connector object, slider value to line
        connector3 = Connector(self.slider2.get_value, fcn3)
        connector4 = Connector(get_b, update_line) # TODO: replace with trigger 

        self.add_updateable_child(connector3)
        self.add_updateable_child(connector4)

    def handle_event(self, event):
        super().handle_event(event)

        self.slider2.handle_event(event)
    




class Level4(ClassificationLevel):
    """
    2x scale and bias (non-linear)
    """
    def init(self):
        # set up this level's unique map
        self.map.reset()

        """ Data """
        xys1 = [
            Point(RED, Vector2(0.5, 0.25), radius = 0.01), 
            Point(RED, Vector2(0.6, 0.2), radius = 0.01), 
        ]
        xys2 = [
            Point(BLUE, Vector2(0.4, 0.6), radius = 0.01), 
            Point(BLUE, Vector2(0.5, 0.75), radius = 0.01), 
            Point(BLUE, Vector2(0.7, 0.8), radius = 0.01), 
            Point(BLUE, Vector2(0.9, 0.4), radius = 0.01), 
        ]
        self.data1 = ClassificationData(xys1, label = "red")
        self.data2 = ClassificationData(xys2, label = "blue")


        xys = xys1 + xys2

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


        # control slider
        slider = Slider(
            xy = Vector2(1.0, 0.5),
            wh = Vector2(1.0, 1.0),
            label = "Scale."
        )
        self.slider = slider

        # add for drawing
        self.map.add_component(slider)

        # response equation
        equation = LinearAffine(0.0, 0.0)
        self.equation = equation


        self.level_success = LevelSuccess1()


        def clicked_cb():
            print("clicked.")
            self.save_if_completed()
                
        # clicker
        clicker = Clicker(
            xy = Vector2(1.5, 1.5),
            wh = Vector2(0.1, 0.1),
            clicked_cb = clicked_cb,
            label = "Check Solution.",
        )
        self.clicker = clicker

        self.map.add_component(clicker)


        
        """ level instructions """
        my_font = pygame.font.SysFont("Arial", 20)
        text_surface = my_font.render("Level 3: Classification. Move the slider to separate the data", True, (255, 255, 255))
        self.map.add_text(text_surface)

        # separator line
        line = Line(SILVER, xy1 = Vector2(0.1, 0.1), xy2 = Vector2(1.1, 0.1), width = 0.05)

        # add for drawing
        self.map.add_line(line)

        ###
        # add for drawing
        self.map.add_component(self.slider)

        # connector1 fcn
        def fcn(a):
            self.equation.a = a * 2.0 - 1.0

        # connector2 fcns
        def get_a():
            return self.equation.a
        
        def update_line(_):
            # update xy1 and xy2
            line.xy1.y = self.equation.forward(line.xy1.x)
            line.xy2.y = self.equation.forward(line.xy2.x)

        # connector object, slider value to line
        connector1 = Connector(self.slider.get_value, fcn)
        connector2 = Connector(get_a, update_line)

        self.add_updateable_child(connector1)
        self.add_updateable_child(connector2)


        ###
        # control slider
        slider2 = Slider(
            xy = Vector2(1.0, 0.25),
            wh = Vector2(1.0, 1.0),
            label = "Bias Value."
        )
        self.slider2 = slider2

        # add for drawing
        self.map.add_component(slider2)

        # connector1 fcn
        def fcn3(b):
            self.equation.b = b * 2.0 - 1.0

        # connector2 fcns
        def get_b():
            return self.equation.b

        # connector object, slider value to line
        connector3 = Connector(self.slider2.get_value, fcn3)
        connector4 = Connector(get_b, update_line) # TODO: replace with trigger 

        self.add_updateable_child(connector3)
        self.add_updateable_child(connector4)

    def g_chart1(self):
        # y1
        chart = Chart(
            Vector2(0.1, 0.1), # location
            Vector2(1.0, 1.0), # scale
        )
        self.chart1 = chart

    def g_chart2(self):
        # y2
        chart = Chart(
            Vector2(0.1, 0.1), # location
            Vector2(1.0, 1.0), # scale
        )
        self.chart2 = chart

    def g_eq2(self):

        # control slider
        slider = Slider(
            xy = Vector2(1.0, 0.5),
            wh = Vector2(1.0, 1.0),
            label = "Scale."
        )
        self.slider = slider

        # add for drawing
        self.map.add_component(slider)

        # response equation
        equation = LinearAffine(0.0, 0.0)
        self.equation = equation


        self.level_success = LevelSuccess1()


        def clicked_cb():
            print("clicked.")
            self.save_if_completed()
                
        # clicker
        clicker = Clicker(
            xy = Vector2(1.5, 1.5),
            wh = Vector2(0.1, 0.1),
            clicked_cb = clicked_cb,
            label = "Check Solution.",
        )
        self.clicker = clicker

        self.map.add_component(clicker)


        
        """ level instructions """
        my_font = pygame.font.SysFont("Arial", 20)
        text_surface = my_font.render("Level 3: Classification. Move the slider to separate the data", True, (255, 255, 255))
        self.map.add_text(text_surface)

        # separator line
        line = Line(SILVER, xy1 = Vector2(0.1, 0.1), xy2 = Vector2(1.1, 0.1), width = 0.05)

        # add for drawing
        self.map.add_line(line)

        ###
        # add for drawing
        self.map.add_component(self.slider)

        # connector1 fcn
        def fcn(a):
            self.equation.a = a * 2.0 - 1.0

        # connector2 fcns
        def get_a():
            return self.equation.a
        
        def update_line(_):
            # update xy1 and xy2
            line.xy1.y = self.equation.forward(line.xy1.x)
            line.xy2.y = self.equation.forward(line.xy2.x)

        # connector object, slider value to line
        connector1 = Connector(self.slider.get_value, fcn)
        connector2 = Connector(get_a, update_line)

        self.add_updateable_child(connector1)
        self.add_updateable_child(connector2)


        ###
        # control slider
        slider2 = Slider(
            xy = Vector2(1.0, 0.25),
            wh = Vector2(1.0, 1.0),
            label = "Bias Value."
        )
        self.slider2 = slider2

        # add for drawing
        self.map.add_component(slider2)

        # connector1 fcn
        def fcn3(b):
            self.equation.b = b * 2.0 - 1.0

        # connector2 fcns
        def get_b():
            return self.equation.b

        # connector object, slider value to line
        connector3 = Connector(self.slider2.get_value, fcn3)
        connector4 = Connector(get_b, update_line) # TODO: replace with trigger 

        self.add_updateable_child(connector3)
        self.add_updateable_child(connector4)


    def handle_event(self, event):
        super().handle_event(event)

        self.slider2.handle_event(event)
    
