# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import List, Tuple, Union, TypeVar

from shapely import affinity
from shapely.geometry import Polygon, Point
from shapely.geometry.base import BaseGeometry

from sketch_utils.sketch_type import AnyLayer, Artboard, Bitmap, Border, Color, Fill, Group, InnerShadow, Page, Shadow, ShapeGroup, \
    SymbolMaster

T = TypeVar('T', bound=AnyLayer)


def recursive_remove_layer(layer: T) -> T:
    """
    Recursively remove all invisible layers from the given layer.
    """
    if isinstance(layer, Page) or isinstance(layer, Artboard) or isinstance(layer, Group) \
            or isinstance(layer, ShapeGroup) or isinstance(layer, SymbolMaster):
        old_layers = layer.layers

        # use to check invisible fill
        def all_alpha_zero(
                lst: Union[List[Fill], List[Border],
                           List[Shadow], List[InnerShadow]]
        ) -> bool:
            return reduce(
                lambda l1, l2: l1 and (l2.color.alpha == 0),
                lst,
                bool(True)
            )

        # Remove all invisible layers
        def need_keep(sub_layer: AnyLayer) -> bool:
            return sub_layer.isVisible

        layer.layers = list(
            # recursive remove current group's all visible layers' children
            map(
                recursive_remove_layer,
                # remove current group's all invisible layers
                filter(
                    need_keep,
                    old_layers
                )
            )
        )
    return layer


@dataclass
class Rotation:
    """
    Rotation of a layer.
    """
    angle: float
    center: Point


@dataclass
class Scale:
    """
    Scale a layer.
    """
    x_scale: float = 1
    y_scale: float = 1
    center: Point = Point(0, 0)

    @classmethod
    def to_flip(cls, horizontal: bool, vertical: bool, center: Point) -> Scale:
        return Scale(-1 if horizontal else 1, -1 if vertical else 1, center)


# all possible transformations
Transform = Union[Rotation, Scale]

# area bounded by mask (already transformed)
BoundArea = BaseGeometry


@dataclass
class LayerTransformed:
    origin: AnyLayer
    bound_area: BoundArea
    opacity: float


def transform_layer(layer: AnyLayer, xy: Point, transforms: Tuple[Transform] = (), bound_areas: Tuple[BoundArea] = (),
                    opacity: float = 1, tint: Color = None) -> LayerTransformed:
    """
    transform the shape
    @param layer: the layer need to be transformed
    @param xy: the absolute axis to the left corner
    @param transforms: the rotation/scale transform list
    @param bound_areas: the mask area to be intersected
    @param opacity: (optional) the opacity of the layer
    @param tint:(optional) the tint of parent layer
    @return:
    """
    # count bounding box's width and height
    width = layer.frame.width
    height = layer.frame.height
    # create a bounding box
    polygon = Polygon(
        ([xy.x, xy.y], [xy.x + width, xy.y], [xy.x + width, xy.y + height], [xy.x, xy.y + height]))
    # execute the transforms
    for transform in reversed(transforms):
        if isinstance(transform, Rotation):
            # use shapely.affinity.rotate to rotate the polygon
            polygon = affinity.rotate(
                polygon, transform.angle, origin=transform.center)
        elif isinstance(transform, Scale):
            # use shapely.affinity.scale to flip the polygon, since scale can be negative
            polygon = affinity.scale(polygon, xfact=transform.x_scale,
                                     yfact=transform.y_scale, origin=transform.center)
    # intersection bounding box with all bound_areas
    for bound_area in bound_areas:
        polygon = polygon.intersection(bound_area)

    # calculate fill color opacity
    layer_opacity = opacity
    if layer.style is not None:
        fill_opacity = 1
        # the most upper layer's tint will replace all the lower layers' tint
        if tint is not None:
            # multiply opacity by tint's alpha
            fill_opacity = tint.alpha
        else:
            # multiply opacity by the style's fill opacity
            if layer.style.fills is not None:
                for fill in layer.style.fills:
                    fill_opacity = fill.color.alpha
        layer_opacity *= fill_opacity
    # return the bounding box
    return LayerTransformed(layer, polygon, layer_opacity)


def get_polygon_exterior(geometry: BaseGeometry) -> List[Tuple[float, float]]:
    """
    get the exterior coord of a polygon
    @param geometry: the BaseGeometry need to be calculated
    @return: the exterior coord of a polygon
    """
    if isinstance(geometry, Polygon) and geometry.exterior is not None and geometry.exterior.coords is not None:
        return list(geometry.exterior.coords)
    return []
