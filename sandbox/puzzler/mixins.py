from pygame.math import Vector2
import nodes

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
    
    def parentscale_to_scale(self, parentxy: Vector2):
        xy = parentxy.elementwise() / (self.wh.elementwise())
        
        return xy
    
    def xy_to_parentxy(self, xy: Vector2):
        """
        transform (scale and offset) xy in this frame to xy w.r.t. my parent
        """
        # multiplication and offset y = ax + b
        parentxy = self.scale_to_parentscale(xy) + self.xy

        return parentxy
    
    def parentxy_to_xy(self, parentxy: Vector2):
        xy = self.parentscale_to_scale(parentxy - self.xy)

        return xy