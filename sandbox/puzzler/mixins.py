from pygame.math import Vector2
import nodes

from screen import Screen

class IDMixin:
    id = 0
    def __init__(self) -> None:
        self.id = IDMixin.id
        IDMixin.id += 1

class FrameMixin:
    """
    A frame, aka a coordinate system
    """
    def __init__(self,
                 xy: Vector2,
                 wh: Vector2,
                 ) -> None:
        self.xy = xy # my xy relative to my parent
        self.wh = wh # my scale relative to my parent's scale


    def scale_to_parentscale(self, xy: Vector2):
        """
        xy is a coordinate in this frame.
        output: xy scaled to the parent scale
        """
        # straight multiplication
        parentxy = xy.elementwise() * (self.wh.elementwise())
        
        return parentxy
    
    def xy_to_parentxy(self, xy: Vector2):
        """
        transform (scale and offset) xy in this frame to xy w.r.t. my parent
        """
        # multiplication and offset y = ax + b
        parentxy = self.scale_to_parentscale(xy) + self.xy

        return parentxy
    
class ScreenMixin:
    def __init__(self) -> None:
        
        # node refs
        assert(isinstance(nodes.SCREEN_NODE, Screen))
        self.screen_node: Screen = nodes.SCREEN_NODE