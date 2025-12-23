import pygame
import random

from pygame.math import Vector2

from structs import Rect, Circle, Line, Point

from mixins import IDMixin, FrameMixin, ScreenMixin

from game import Game

from iomixins import IOMixin

from constants import *

from utils import get_random_rgb

from screen import Screen

from components import Slider, Chart

from levels import Level, Level1

import nodes

from io_ import PlayerInputs





def main():

    # spool up global level class instances
    pygame.init()

    # nodes
    nodes.SCREEN_NODE = Screen()
    nodes.IO_NODE = PlayerInputs()

    # instantiate the game
    nodes.GAME_NODE = Game()

    # load a level
    nodes.GAME_NODE.load_level(Level1)

    # run the game loop
    nodes.GAME_NODE.run()




if __name__ == "__main__":
    main()