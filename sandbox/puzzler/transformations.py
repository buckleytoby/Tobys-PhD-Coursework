

from structs import Rect, Circle

from mixins import FrameMixin

    
def rect_in_parent_frame(rect: Rect, frame: FrameMixin):
    copy = rect.copy()

    copy.xy = frame.xy_to_parentxy(rect.xy)

    # TODO: scale wh
    copy.wh = frame.scale_to_parentscale(rect.wh)

    return copy
    
def circle_in_parent_frame(circle: Circle, frame: FrameMixin):
    copy = circle.copy()

    copy.xy = frame.xy_to_parentxy(circle.xy)

    # TODO: scale radius

    return copy