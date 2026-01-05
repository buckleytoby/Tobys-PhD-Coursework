import math

import pygame
from pygame.math import Vector2

import nodes

from assets import Assets


class Sprite(pygame.sprite.Sprite):
    def __init__(self,
                 asset_key = None,
                 xy = Vector2(0, 0),
                 wh = Vector2(1, 1),
                 ):
        pygame.sprite.Sprite.__init__(self)

        self.xy = xy
        self.wh = wh
        self.original_wh = self.wh.copy()


        # get ref
        assert(nodes.ASSETS_NODE is not None)
        assets: Assets = nodes.ASSETS_NODE
        
        # make a local mutable copy of my desired asset
        if asset_key is not None:
            self.image = assets[asset_key].copy()

class StretchAndSqueeze(Sprite):
    def update(self):
        """ Scale in pixel units """
        # 
        gt = nodes.GAME_TIME

        # 10% stretch/squeeze, centered on 1.0
        mx = math.cos(gt) * 0.1 + 1.0
        my = math.sin(gt) * 0.1 + 1.0

        mxy = Vector2(mx, my)

        self.wh = self.original_wh.elementwise() * mxy.elementwise()

