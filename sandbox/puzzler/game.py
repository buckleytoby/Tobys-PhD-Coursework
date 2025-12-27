from mixins import ScreenMixin

from iomixins import IOMixin

import pygame

from levels.levels import *

import sys

## parent classes
class Base:
    def run(self):
        pass

## end parent classes


class Game(Base, IOMixin, ScreenMixin):
    """
    Top-level 7 things
    """
    def __init__(self,
                 dt = 1/60,
                 fps = 60,
                 ) -> None:
        IOMixin.__init__(self)
        ScreenMixin.__init__(self)

        # my parameters
        self.dt = dt
        self.fps = fps
        
        # my <=7 members
        self.clock = pygame.time.Clock()
        self.active_level = None
        self.level_yielder = self.get_next_level()

        # node refs

    def draw(self):
        # draw the screen
        self.screen_node.draw()

        # draw the level
        if isinstance(self.active_level, Level):
            self.active_level.draw()

    def update(self):
        # update the level
        assert(self.active_level is not None)
        self.active_level.update()

    def handle_event(self, event):
        # self.active_level: Level
        assert(isinstance(self.active_level, Level))
        self.active_level.handle_event(event)

    
    def process_some_events(self):
        # TODO: move to separate event handlers?
        for event in pygame.event.get():
            self.handle_event(event)

    def end_game(self):
        print("No more levels.")
        print("You win!")

        pygame.quit()
        sys.exit()

    def get_next_level(self):
        levels = [
            TestLevel1,
            Level1,
            Level2,
            Level3,
        ]

        for level in levels:
            yield level

        self.end_game()

    def progress(self):
        assert(self.active_level is not None)

        if self.active_level.is_completed:
            print("level complete")
            self.load_level()

    def step(self):
        # process some events
        self.process_some_events()

        # update the game state
        self.update()

        # draw
        self.draw()

        # render
        self.screen_node.display()

        # check progress
        self.progress()

        # tick
        self.clock.tick(self.fps)


    def run(self):
        """
        infinite game loop
        """
        done = False
        while not done:
            self.step()

            pass

    def load_level(self, level_class = None):
        if level_class is None:
            level_class = next(self.level_yielder)
            
        # make an instance
        lvl: Level = level_class()

        # save it
        self.active_level = lvl
