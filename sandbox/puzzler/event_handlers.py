from iomixins import IOMixin

import pygame


class ClickAndDrag(IOMixin):
    """
    
    """
    def __init__(self) -> None:
        IOMixin.__init__(self)

    def handle_event(self, event: pygame.event.Event, handle, handle_radius):
        """Processes mouse clicks and releases."""
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = self.io_node.get_mouse_pos()

            # Check if mouse is over the handle
            dist = mouse_pos.distance_to(handle.xy)
            if dist < handle_radius:
                self.dragging = True
                
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False