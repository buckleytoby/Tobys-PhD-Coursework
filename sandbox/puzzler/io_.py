import pygame
from pygame.math import Vector2

import nodes


from screen import Screen

class PlayerInputs:
    """
    Interface to pygame I/O, returns values in game frame
    """
    def __init__(self) -> None:
        
        self.mapping = {}

        # load default mapping
        self.default_mapping()

        # node references
        assert(isinstance(nodes.SCREEN_NODE, Screen))
        self.screen_node: Screen = nodes.SCREEN_NODE


    def default_mapping(self):
        self.mapping = {
            "right": pygame.K_RIGHT,
        }

    def get_pygame_key(self, map_key):
        return self.mapping[map_key]

    def is_pressed(self, map_key):
        pygame_key = self.get_pygame_key(map_key)

    def is_pygame_pressed(self, pygame_key):
        keys = pygame.key.get_pressed()

        if pygame_key in keys:
            return True
        else:
            return False

    def get_mouse_pos(self):
        pygame_xy = Vector2(pygame.mouse.get_pos())

        xy = self.screen_node.pygamexy_to_xy(pygame_xy, 0)

        return xy
    
    

# class EventHandler:
#     def __init__(self) -> None:
        
#         # node refs
#         assert(isinstance(nodes.IO, PlayerInputs))
#         self.player_inputs_node: PlayerInputs = nodes.PLAYER_INPUTS

#     def handle_event(self, event: pygame.event.Event):
#         pass