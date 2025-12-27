from iomixins import IOMixin

import pygame

from structs import Rect


class Click(IOMixin):
    """
    
    """
    def __init__(self) -> None:
        IOMixin.__init__(self)

    def handle_event(self, event: pygame.event.Event, state, rect_global: Rect):
        """Processes mouse clicks and releases."""
                
        if event.type == pygame.MOUSEBUTTONDOWN:
            # this is global
            mouse_pos = self.io_node.get_mouse_pos()

            # Check if mouse is over the handle
            inside = rect_global.get_pygame_rect().collidepoint(mouse_pos)

            if inside:
                state.clicked = True

        if event.type == pygame.MOUSEBUTTONUP:
            state.clicked = False


class ClickAndDrag(IOMixin):
    """
    
    """
    def __init__(self) -> None:
        IOMixin.__init__(self)

    def handle_event(self, event: pygame.event.Event, state, handle_parentxy, handle_radius):
        """Processes mouse clicks and releases."""
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # this is global
            mouse_pos = self.io_node.get_mouse_pos()

            # Check if mouse is over the handle
            dist = mouse_pos.distance_to(handle_parentxy)
            if dist < handle_radius:
                state.dragging = True
                
        if event.type == pygame.MOUSEBUTTONUP:
            state.dragging = False
