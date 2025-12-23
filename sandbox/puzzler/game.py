from mixins import ScreenMixin

from iomixins import IOMixin

import pygame

from levels import Level

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

    def step(self):
        # process some events
        self.process_some_events()

        # update the game state
        self.update()

        # draw
        self.draw()

        # render
        self.screen_node.display()

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

    def load_level(self, level_class):
        # make an instance
        lvl = level_class()

        # save it
        self.active_level = lvl
