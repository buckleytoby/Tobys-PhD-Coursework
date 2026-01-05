import pygame
import random

from pygame.math import Vector2

from structs import Rect, Circle, Line, Point

from mixins import IDMixin, FrameMixin

from game import Game

from iomixins import IOMixin

from constants import *

from utils import get_random_rgb

from screen import Screen, ScreenMixin

from components import Slider, Chart

from levels.levels import *

import nodes

from io_ import PlayerInputs

from assets import Assets





def main():

    # spool up global level class instances
    pygame.init()

    ## nodes
    nodes.SCREEN_NODE = Screen()
    nodes.IO_NODE = PlayerInputs()
    nodes.ASSETS_NODE = Assets()
    ##

    # instantiate the game
    nodes.GAME_NODE = Game()

    # load the default level
    nodes.GAME_NODE.load_level()

    ### run the game loop
    nodes.GAME_NODE.run()




if __name__ == "__main__":
    main()