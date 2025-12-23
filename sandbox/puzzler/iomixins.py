
from io_ import PlayerInputs

import nodes
    
class IOMixin:
    def __init__(self) -> None:
        
        # node refs
        assert(isinstance(nodes.IO_NODE, PlayerInputs))
        self.io_node: PlayerInputs = nodes.IO_NODE


class EventHandlerMixin(IOMixin):
    def handle_event(self, event):
        pass