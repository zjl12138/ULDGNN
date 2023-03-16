# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Tuple

from fastclasses_json import JSONMixin, dataclass_json


def to_sketch_object(obj: Dict[str, Any]) -> Optional[SketchObject]:
    if (obj is not None) and ('_class' in obj.keys()) and (obj['_class'] in CLASS_MAP.keys()):
        return CLASS_MAP[obj['_class']].from_dict(obj)
    else:
        return None


@dataclass_json
@dataclass
class SketchObject(JSONMixin):
    class_: str = field(metadata={
        "fastclasses_json": {
            "field_name": "_class"
        }
    })


@dataclass_json
@dataclass
class LayerObjectRequired(JSONMixin):
    do_objectID: Uuid
    booleanOperation: BooleanOperation
    exportOptions: ExportOptions
    frame: Rect
    isFixedToViewport: bool
    isFlippedHorizontal: bool
    isFlippedVertical: bool
    isLocked: bool
    isVisible: bool
    layerListExpandedType: LayerListExpanded
    name: str
    nameIsFixed: bool
    resizingConstraint: float
    resizingType: ResizeType
    rotation: float
    shouldBreakMaskChain: bool

    def find_layer_by_id(self, obj_id: str, key_stack: Tuple[str] = ()) -> Optional[List[str]]:
        """
        find the path of the target layer
        @param obj_id: the target layer's do_objectID
        @param key_stack: the layer name stack passing through
        @return: the list of layer stack until the target layer 
        """
        if self.do_objectID == obj_id:
            return list(key_stack + (self.name,))
        elif hasattr(self, 'layers'):
            for sub_layer in self.layers:
                result = sub_layer.find_layer_by_id(
                    obj_id, key_stack + (self.name,))
                if result is not None:
                    return result
        return None


@dataclass_json
@dataclass
class LayerObjectOptional(JSONMixin):
    flow: Optional[FlowConnection] = None
    sharedStyleID: Optional[Uuid] = None
    hasClippingMask: Optional[bool] = None
    clippingMaskMode: Optional[int] = None
    userInfo: Optional[Any] = None
    style: Optional[Style] = None
    maintainScrollPosition: Optional[bool] = None


@dataclass_json
@dataclass
class LayerGroupRequired(LayerObjectRequired):
    hasClickThrough: bool
    layers: List[Union[Group, Oval, Polygon, Rectangle, ShapePath, Star,
                       Triangle, ShapeGroup, Text, SymbolInstance, Slice, Hotspot, Bitmap]] = field(
        metadata={
            "fastclasses_json": {
                "decoder": lambda lst: list(
                    filter(
                        lambda x: x is not None,
                        map(to_sketch_object, lst)
                    )
                )
            }
        })


@dataclass_json
@dataclass
class LayerGroupOptional(LayerObjectOptional):
    groupLayout: Optional[Union[FreeformGroupLayout,
                                InferredGroupLayout]] = field(
        default=None,
        metadata={
            "fastclasses_json": {
                "decoder": to_sketch_object
            }
        })


# UUID string.
Uuid = str


# Collection of global document objects
@dataclass_json
@dataclass
class AssetCollection(SketchObject):
    do_objectID: Uuid
    colorAssets: List[ColorAsset]
    gradientAssets: List[GradientAsset]
    images: List[Union[FileRef, DataRef]] = field(metadata={
        "fastclasses_json": {
            "decoder": lambda lst: list(
                filter(
                    lambda x: x is not None,
                    map(to_sketch_object, lst)
                )
            )
        }
    })
    colors: List[Color]
    gradients: List[Gradient]
    exportPresets: List[Any]
    imageCollection: Optional[ImageCollection] = None


# Legacy object only retained for migrating older documents.
@dataclass_json
@dataclass
class ImageCollection(SketchObject):
    images: Any


# Defines a reusable color asset
@dataclass_json
@dataclass
class ColorAsset(SketchObject):
    do_objectID: Uuid
    name: str
    color: Color


# Defines a RGBA color value
@dataclass_json
@dataclass
class Color(SketchObject):
    alpha: UnitInterval
    blue: UnitInterval
    green: UnitInterval
    red: UnitInterval
    swatchID: Optional[Uuid] = None


# The set of all real numbers that are greater than or equal to 0 and less than or equal to 1. Used within Sketch
# documents to encode normalised scalar values, for example RGB color components.
UnitInterval = float


# Defines a reusable gradient asset
@dataclass_json
@dataclass
class GradientAsset(SketchObject):
    do_objectID: Uuid
    name: str
    gradient: Gradient


# Defines a gradient
@dataclass_json
@dataclass
class Gradient(SketchObject):
    elipseLength: float
    gradientType: GradientType
    stops: List[GradientStop]
    to: str
    from_: str = field(metadata={
        "fastclasses_json": {
            "field_name": "from"
        }
    })


# Enumeration of the gradient types
class GradientType(Enum):
    Linear = 0
    Radial = 1
    Angular = 2


# A formatted string representation of a 2D point, e.g. {1, 1}.
PointString = str


@dataclass_json
@dataclass
class GradientStop(SketchObject):
    color: Color
    position: UnitInterval


# Defines a reference to a file within the document bundle
@dataclass_json
@dataclass
class FileRef(SketchObject):
    _ref_class: Union[Literal['MSImageData'],
                      Literal['MSImmutablePage'],
                      Literal['MSPatch']]
    _ref: str


# Defines inline base64 data
@dataclass_json
@dataclass
class DataRef(SketchObject):
    _ref_class: Union[Literal['MSImageData'], Literal['MSFontData']]
    _ref: str
    data: Base64Data
    sha1: Base64Data


@dataclass_json
@dataclass
class Base64Data(JSONMixin):
    data: str = field(metadata={
        "fastclasses_json": {
            "field_name": "_data"
        }
    })


# Enumeration of the color profiles Sketch can use to render a document
class ColorSpace:
    Unmanaged = 0
    SRGB = 1
    P3 = 2


# Defines a layer style that has been imported from a library
@dataclass_json
@dataclass
class ForeignLayerStyle(SketchObject):
    do_objectID: Uuid
    libraryID: Uuid
    sourceLibraryName: str
    symbolPrivate: bool
    remoteStyleID: Uuid
    localSharedStyle: SharedStyle


# Defines a reusable style
@dataclass_json
@dataclass
class SharedStyle(SketchObject):
    do_objectID: Uuid
    name: str
    value: Style


# Defines a layer style
@dataclass_json
@dataclass
class Style(SketchObject):
    do_objectID: str
    borderOptions: BorderOptions
    startMarkerType: MarkerType
    endMarkerType: MarkerType
    miterLimit: float
    windingRule: WindingRule
    colorControls: ColorControls
    borders: Optional[List[Border]] = None
    blur: Optional[Blur] = None
    fills: Optional[List[Fill]] = None
    textStyle: Optional[TextStyle] = None
    shadows: Optional[List[Shadow]] = None
    innerShadows: Optional[List[InnerShadow]] = None
    contextSettings: Optional[GraphicsContextSettings] = None


# Defines a border style
@dataclass_json
@dataclass
class Border(SketchObject):
    isEnabled: bool
    color: Color
    fillType: FillType
    position: BorderPosition
    thickness: float
    contextSettings: GraphicsContextSettings
    gradient: Gradient


# Enumeration of the fill types
class FillType(Enum):
    Color = 0
    Gradient = 1
    Pattern = 4


# Enumeration of border positions
class BorderPosition(Enum):
    Center = 0
    Inside = 1
    Outside = 2


# Defines the opacity and blend mode of a style or shadow
@dataclass_json
@dataclass
class GraphicsContextSettings(SketchObject):
    blendMode: BlendMode
    opacity: float


# Enumeration of the blend modes that can be applied to fills
class BlendMode(Enum):
    Normal = 0
    Darken = 1
    Multiply = 2
    ColorBurn = 3
    Lighten = 4
    Screen = 5
    ColorDodge = 6
    Overlay = 7
    SoftLight = 8
    HardLight = 9
    Difference = 10
    Exclusion = 11
    Hue = 12
    Saturation = 13
    Color = 14
    Luminosity = 15
    PlusDarker = 16
    PlusLighter = 17

    # happen in some old sketches
    UndefinedMode1 = 232


# Defines border options
@dataclass_json
@dataclass
class BorderOptions(SketchObject):
    isEnabled: bool
    dashPattern: List[float]
    lineCapStyle: LineCapStyle
    lineJoinStyle: LineJoinStyle


# Enumeration of the line cap styles
class LineCapStyle(Enum):
    Butt = 0
    Round = 1
    Projecting = 2


# Enumeration of the line join styles
class LineJoinStyle(Enum):
    Miter = 0
    Round = 1
    Bevel = 2


# Defines a blur style
@dataclass_json
@dataclass
class Blur(SketchObject):
    isEnabled: bool
    center: PointString
    saturation: float
    type: BlurType
    motionAngle: Optional[float] = None
    radius: Optional[float] = None


# Enumeration of the various blur types
class BlurType(Enum):
    Gaussian = 0
    Motion = 1
    Zoom = 2
    Background = 3


# Defines a fill style
@dataclass_json
@dataclass
class Fill(SketchObject):
    isEnabled: bool
    color: Color
    fillType: FillType
    noiseIndex: float
    noiseIntensity: float
    patternFillType: PatternFillType
    patternTileScale: float
    contextSettings: GraphicsContextSettings
    gradient: Gradient
    image: Optional[Union[FileRef, DataRef]] = field(
        default=None,
        metadata={
            "fastclasses_json": {
                "decoder": to_sketch_object
            }
        }
    )


# Enumeration of pattern fill types
class PatternFillType(Enum):
    Tile = 0
    Fill = 1
    Stretch = 2
    Fit = 3


# Enumeration of the possible types of vector line endings
class MarkerType(Enum):
    OpenArrow = 0
    FilledArrow = 1
    Line = 2
    OpenCircle = 3
    FilledCircle = 4
    OpenSquare = 5
    FilledSquare = 6


# Enumeration of the winding rule that controls how fills behave in shapes with complex paths
class WindingRule(Enum):
    NonZero = 0
    EvenOdd = 1


# Defines text style
@dataclass_json
@dataclass
class TextStyle(SketchObject):
    verticalAlignment: TextVerticalAlignment
    encodedAttributes: EncodedAttributes


# Defines encoded attributes
@dataclass_json
@dataclass
class EncodedAttributes(JSONMixin):
    MSAttributedStringFontAttribute: FontDescriptor
    paragraphStyle: Optional[ParagraphStyle] = None
    MSAttributedStringTextTransformAttribute: Optional[TextTransform] = None
    underlineStyle: Optional[UnderlineStyle] = None
    strikethroughStyle: Optional[Any] = None
    kerning: Optional[float] = None
    textStyleVerticalAlignmentKey: Optional[TextVerticalAlignment] = None
    MSAttributedStringColorAttribute: Optional[Color] = None


# Enumeration of the text style vertical alighment options
class TextVerticalAlignment(Enum):
    Top = 0
    Middle = 1
    Bottom = 2


# Defines the paragraph style within a text style
@dataclass_json
@dataclass
class ParagraphStyle(SketchObject):
    alignment: Optional[TextHorizontalAlignment] = None
    baseWritingDirection: Optional[float] = None
    maximumLineHeight: Optional[float] = None
    minimumLineHeight: Optional[float] = None
    allowsDefaultTighteningForTruncation: Optional[float] = None


# Enumeration of the horizontal alignment options for paragraphs
class TextHorizontalAlignment(Enum):
    Left = 0
    Right = 1
    Centered = 2
    Justified = 3
    Natural = 4


# Enumeration of the text style transformations options
class TextTransform(Enum):
    None_ = 0
    Uppercase = 1
    Lowercase = 2


# Enumeration of the text style underline options
class UnderlineStyle(Enum):
    None_ = 0
    Underlined = 1


# Defines a font selection
@dataclass_json
@dataclass
class FontDescriptor(SketchObject):
    attributes: FontAttributes


# Defines a font attribute
@dataclass_json
@dataclass
class FontAttributes(SketchObject):
    name: str
    size: float
    variation: Optional[Dict[str, float]] = None


# Defines a shadow style
@dataclass_json
@dataclass
class Shadow(SketchObject):
    isEnabled: bool
    blurRadius: float
    color: Color
    contextSettings: GraphicsContextSettings
    offsetX: float
    offsetY: float
    spread: float


# Defines an inner shadow style
@dataclass_json
@dataclass
class InnerShadow(SketchObject):
    isEnabled: bool
    blurRadius: float
    color: Color
    contextSettings: GraphicsContextSettings
    offsetX: float
    offsetY: float
    spread: float


# Defines color adjust styles on images
@dataclass_json
@dataclass
class ColorControls(SketchObject):
    isEnabled: bool
    brightness: float
    contrast: float
    hue: float
    saturation: float


# Defines a symbol that has been imported from a library
@dataclass_json
@dataclass
class ForeignSymbol(SketchObject):
    do_objectID: Uuid
    libraryID: Uuid
    sourceLibraryName: str
    symbolPrivate: bool
    originalMaster: SymbolMaster
    symbolMaster: SymbolMaster
    missingLibraryFontAcknowledged: Optional[bool] = None


# A symbol source layer represents a reusable group of layers
@dataclass_json
@dataclass
class SymbolMasterContent(JSONMixin):
    horizontalRulerData: RulerData
    verticalRulerData: RulerData
    backgroundColor: Color
    hasBackgroundColor: bool
    includeBackgroundColorInInstance: bool
    includeBackgroundColorInExport: bool
    isFlowHome: bool
    resizesContent: bool
    symbolID: Uuid
    allowsOverrides: bool
    overrideProperties: List[OverrideProperty]
    layout: Optional[LayoutGrid] = None
    grid: Optional[SimpleGrid] = None
    presetDictionary: Optional[Any] = None


@dataclass_json
@dataclass
class SymbolMaster(LayerGroupOptional, SymbolMasterContent, LayerGroupRequired, SketchObject):
    pass


# Enumeration of the boolean operations that can be applied to combine shapes
class BooleanOperation(Enum):
    None_ = -1
    Union = 0
    Subtract = 1
    Intersection = 2
    Difference = 3


# Enumeration of the boolean operations that can be applied to combine shapes
@dataclass_json
@dataclass
class ExportOptions(SketchObject):
    exportFormats: List[ExportFormat]
    includedLayerIds: List[Uuid]
    layerOptions: float
    shouldTrim: bool


# Defines an export format, as listed in a layer's export options
@dataclass_json
@dataclass
class ExportFormat(SketchObject):
    absoluteSize: float
    fileFormat: ExportFileFormat
    name: str
    scale: float
    visibleScaleType: VisibleScaleType
    namingScheme: Optional[ExportFormatNamingScheme] = None


# Enumeration of the file formats that can be selected in the layer export options
class ExportFileFormat(Enum):
    PNG = 'png'
    JPG = 'jpg'
    TIFF = 'tiff'
    EPS = 'eps'
    PDF = 'pdf'
    WEBP = 'webp'
    SVG = 'svg'


# Enumeration of the possible types of export format naming schemes
class ExportFormatNamingScheme(Enum):
    Suffix = 0
    Prefix = 1


# Enumeration of the possible values to control how an exported layer will be scaled
class VisibleScaleType(Enum):
    Scale = 0
    Width = 1
    Height = 2


# Defines an abstract rectangle
@dataclass_json
@dataclass
class Rect(SketchObject):
    constrainProportions: bool
    height: float
    width: float
    x: float
    y: float


# Defines a connection between elements in a prototype
@dataclass_json
@dataclass
class FlowConnection(SketchObject):
    destinationArtboardID: Union[Uuid, Literal['back']]
    animationType: AnimationType
    maintainScrollPosition: Optional[bool] = None


# Enumeration of the animation transition types between prototype screens
class AnimationType(Enum):
    # happen in some old sketches
    Undefined = -1
    None_ = 0
    SlideFromLeft = 1
    SlideFromRight = 2
    SlideFromBottom = 3
    SlideFromTop = 4


# Enumeration of the expansion states in the layer list UI
class LayerListExpanded(Enum):
    Undecided = 0
    Collapsed = 1
    Expanded = 2


# Enumeration of the possible resize types
class ResizeType(Enum):
    Stretch = 0
    PinToEdge = 1
    Resize = 2
    Float = 3


# Defines persisted ruler positions on artboards, pages and symbols
@dataclass_json
@dataclass
class RulerData(SketchObject):
    base: float
    guides: List[float]


# Defines the layout settings for an artboard or page
@dataclass_json
@dataclass
class LayoutGrid(SketchObject):
    isEnabled: bool
    columnWidth: float
    gutterHeight: float
    gutterWidth: float
    horizontalOffset: float
    numberOfColumns: float
    rowHeightMultiplication: float
    totalWidth: float
    guttersOutside: bool
    drawHorizontal: bool
    drawHorizontalLines: bool
    drawVertical: bool


# Defines the grid settings for an artboard or page
@dataclass_json
@dataclass
class SimpleGrid(SketchObject):
    isEnabled: bool
    gridSize: float
    thickGridTimes: float


# Normal group layout
FreeformGroupLayout = SketchObject


# Inferred group layout defines smart layout options
@dataclass_json
@dataclass
class InferredGroupLayout(SketchObject):
    axis: InferredLayoutAxis
    layoutAnchor: InferredLayoutAnchor
    maxSize: Optional[float] = None
    minSize: Optional[float] = None


# Enumeration of the axis types for inferred (aka smart) layout
class InferredLayoutAxis(Enum):
    Horizontal = 0
    Vertical = 1


# Enumeration of the anchor types for inferred (aka smart) layout
class InferredLayoutAnchor(Enum):
    Min = 0
    Middle = 1
    Max = 2


# Defines override properties on symbol sources
@dataclass_json
@dataclass
class OverrideProperty(SketchObject):
    overrideName: OverrideName
    canOverride: bool


# Defines the valid string patterns for an override name
OverrideName = str


# Group layers are a document organisation aid
@dataclass_json
@dataclass
class Group(LayerGroupOptional, LayerGroupRequired, SketchObject):
    pass


# Oval layers are the result of adding an oval shape to the canvas
@dataclass_json
@dataclass
class OvalContent(JSONMixin):
    edited: bool
    isClosed: bool
    pointRadiusBehaviour: PointsRadiusBehaviour
    points: List[CurvePoint]


@dataclass_json
@dataclass
class Oval(LayerObjectOptional, OvalContent, LayerObjectRequired, SketchObject):
    pass


# Enumeration of the possible values for corner rounding on shape points.
class PointsRadiusBehaviour(Enum):
    Disabled = -1
    Legacy = 0
    Rounded = 1
    Smooth = 2


# Defines a shape layer curve point
@dataclass_json
@dataclass
class CurvePoint(SketchObject):
    cornerRadius: float
    curveFrom: PointString
    curveTo: PointString
    hasCurveFrom: bool
    hasCurveTo: bool
    curveMode: CurveMode
    point: PointString


# Enumeration of the curve modes that can be applied to vector points
class CurveMode(Enum):
    None_ = 0
    Straight = 1
    Mirrored = 2
    Asymmetric = 3
    Disconnected = 4


# Polygon layers are the result of adding a polygon shape to the canvas
@dataclass_json
@dataclass
class PolygonContent(JSONMixin):
    edited: bool
    isClosed: bool
    pointRadiusBehaviour: PointsRadiusBehaviour
    points: List[CurvePoint]
    numberOfPoints: int


@dataclass_json
@dataclass
class Polygon(LayerObjectOptional, PolygonContent, LayerObjectRequired, SketchObject):
    pass


# Rectangle layers are the result of adding a rectangle shape to the canvas
@dataclass_json
@dataclass
class RectangleContent(JSONMixin):
    edited: bool
    isClosed: bool
    pointRadiusBehaviour: PointsRadiusBehaviour
    points: List[CurvePoint]
    fixedRadius: float
    hasConvertedToNewRoundCorners: bool
    needsConvertionToNewRoundCorners: bool


@dataclass_json
@dataclass
class Rectangle(LayerObjectOptional, RectangleContent, LayerObjectRequired, SketchObject):
    pass


# Shape path layers are the result of adding a vector layer
@dataclass_json
@dataclass
class ShapePathContent(JSONMixin):
    edited: bool
    isClosed: bool
    pointRadiusBehaviour: PointsRadiusBehaviour
    points: List[CurvePoint]


@dataclass_json
@dataclass
class ShapePath(LayerObjectOptional, ShapePathContent, LayerObjectRequired, SketchObject):
    pass


# Star layers are the result of adding a star shape to the canvas
@dataclass_json
@dataclass
class StarContent(JSONMixin):
    edited: bool
    isClosed: bool
    pointRadiusBehaviour: PointsRadiusBehaviour
    points: List[CurvePoint]
    numberOfPoints: int
    radius: float


@dataclass_json
@dataclass
class Star(LayerObjectOptional, StarContent, LayerObjectRequired, SketchObject):
    pass


# Triangle layers are the result of adding a triangle shape to the canvas
@dataclass_json
@dataclass
class TriangleContent(JSONMixin):
    edited: bool
    isClosed: bool
    pointRadiusBehaviour: PointsRadiusBehaviour
    points: List[CurvePoint]
    isEquilateral: bool


@dataclass_json
@dataclass
class Triangle(LayerObjectOptional, TriangleContent, LayerObjectRequired, SketchObject):
    pass


# Shape groups layers group together multiple shape layers
@dataclass_json
@dataclass
class ShapeGroupContent(JSONMixin):
    windingRule: WindingRule


@dataclass_json
@dataclass
class ShapeGroup(LayerGroupOptional, ShapeGroupContent, LayerGroupRequired, SketchObject):
    pass


# A text layer represents a discrete block or line of text
@dataclass_json
@dataclass
class TextContent(JSONMixin):
    attributedString: AttributedString
    automaticallyDrawOnUnderlyingPath: bool
    dontSynchroniseWithSymbol: bool
    lineSpacingBehaviour: LineSpacingBehaviour
    textBehaviour: TextBehaviour
    glyphBounds: PointListString


@dataclass_json
@dataclass
class Text(LayerObjectOptional, TextContent, LayerObjectRequired, SketchObject):
    pass


# Defines character strings and associated styling applied to character ranges
@dataclass_json
@dataclass
class AttributedString(SketchObject):
    string: str
    attributes: List[StringAttribute]


# Defines an attribute assigned to a range of characters in an attributed string
@dataclass_json
@dataclass
class StringAttribute(SketchObject):
    location: float
    length: int
    attributes: StringAttributeAttributes


@dataclass_json
@dataclass
class StringAttributeAttributes(JSONMixin):
    MSAttributedStringFontAttribute: FontDescriptor
    kerning: Optional[float] = None
    textStyleVerticalAlignmentKey: Optional[TextVerticalAlignment] = None
    MSAttributedStringColorAttribute: Optional[Color] = None
    paragraphStyle: Optional[ParagraphStyle] = None


# Enumeration of line spacing behaviour for fixed line height text
class LineSpacingBehaviour(Enum):
    None_ = 0
    Legacy = 1
    ConsistentBaseline = 2
    # happen in some old sketches
    Undefined = 3


# Enumeration of the behaviours for text layers
class TextBehaviour(Enum):
    Flexible = 0
    Fixed = 1
    FixedWidthAndHeight = 2


# A string representation of a series of 2D points, in the format {{x, y}, {x,y}}.
PointListString = str


# Symbol instance layers represent an instance of a symbol source
@dataclass_json
@dataclass
class SymbolInstanceContent(JSONMixin):
    overrideValues: List[OverrideValue]
    scale: float
    symbolID: Uuid
    verticalSpacing: float
    horizontalSpacing: float


@dataclass_json
@dataclass
class SymbolInstance(LayerObjectOptional, SymbolInstanceContent, LayerObjectRequired, SketchObject):
    pass


# Defines an individual symbol override
@dataclass_json
@dataclass
class OverrideValue(SketchObject):
    overrideName: OverrideName
    value: Union[str, Uuid, FileRef, DataRef] = field(metadata={
        "fastclasses_json": {
            "decoder": lambda x: x if isinstance(x, str) else to_sketch_object(x)
        }
    })
    do_objectID: Optional[Uuid] = None


# Slice layers allow the content beneath their frame to be exported
@dataclass_json
@dataclass
class SliceContent(JSONMixin):
    hasBackgroundColor: bool
    backgroundColor: Color


@dataclass_json
@dataclass
class Slice(LayerObjectOptional, SliceContent, LayerObjectRequired, SketchObject):
    pass


# Hotspot layers define clickable hotspots for use with prototypes
@dataclass_json
@dataclass
class Hotspot(LayerObjectOptional, LayerObjectRequired, SketchObject):
    pass


# Bitmap layers house a single image
@dataclass_json
@dataclass
class BitmapContent(JSONMixin):
    fillReplacesImage: bool
    image: Union[FileRef, DataRef] = field(metadata={
        "fastclasses_json": {
            "decoder": to_sketch_object
        }
    })
    intendedDPI: float
    clippingMask: PointListString


@dataclass_json
@dataclass
class Bitmap(LayerObjectOptional, BitmapContent, LayerObjectRequired, SketchObject):
    pass


# Defines a text style that has been imported from a library
@dataclass_json
@dataclass
class ForeignTextStyle(SketchObject):
    libraryID: Uuid
    sourceLibraryName: str
    symbolPrivate: bool
    remoteStyleID: Uuid
    localSharedStyle: SharedStyle
    missingLibraryFontAcknowledged: Optional[bool] = None


# Defines a swatch that has been imported from a library
@dataclass_json
@dataclass
class ForeignSwatch(SketchObject):
    do_objectID: Uuid
    libraryID: Uuid
    sourceLibraryName: str
    symbolPrivate: bool
    remoteSwatchID: Uuid
    localSwatch: Swatch


# Defines a swatch color variable.
@dataclass_json
@dataclass
class Swatch(SketchObject):
    do_objectID: Uuid
    name: str
    value: Color


# Defines a document's list of reusable styles
@dataclass_json
@dataclass
class SharedStyleContainer(SketchObject):
    objects: List[SharedStyle]
    do_objectID: Optional[Uuid] = None


# Defines a document's list of reusable text styles
@dataclass_json
@dataclass
class SharedTextStyleContainer(SketchObject):
    objects: List[SharedStyle]
    do_objectID: Optional[Uuid] = None


# Legacy object only retained for migrating older documents.
@dataclass_json
@dataclass
class SymbolContainer(SketchObject):
    objects: List[Any]
    do_objectID: Optional[Uuid] = None


# Defines a document's list of swatches
@dataclass_json
@dataclass
class SwatchContainer(SketchObject):
    objects: List[Swatch]
    do_objectID: Optional[Uuid] = None


# Defines a document's list of swatches
@dataclass_json
@dataclass
class FontRef(SketchObject):
    fontData: DataRef
    fontFamilyName: str
    fontFileName: str
    options: float
    postscriptNames: List[str]


# Container for ephemeral document state. For now this is just a placeholder, and will see additions in future
# document versions.
DocumentState = Any


# Defines ephemeral patch information related to the Cloud collaborative editing feature. This information will only
# be found behind-the-scenes in Cloud documents and won't be relevant or visible to users parsing or generating their
# own Sketch documents.
@dataclass_json
@dataclass
class PatchInfo(SketchObject):
    baseVersionID: Uuid
    lastIntegratedPatchID: Uuid
    localPatches: List[FileRef]
    receivedPatches: List[FileRef]


# Page layers are the top level organisational abstraction within a document
@dataclass_json
@dataclass
class PageContent(JSONMixin):
    horizontalRulerData: RulerData
    verticalRulerData: RulerData
    layout: Optional[LayoutGrid] = None
    grid: Optional[SimpleGrid] = None


@dataclass_json
@dataclass
class Page(LayerGroupOptional, PageContent, LayerGroupRequired, SketchObject):
    @classmethod
    def from_json(cls, json_data: Union[str, bytes]) -> Page:
        return super().from_json(json_data)

    @classmethod
    def from_dict(cls, o: dict) -> Page:
        return super().from_dict(o)


# Artboard layers are a document organisation aid. They have a fixed frame that usually map to variations of device
# dimensions or viewport sizes
@dataclass_json
@dataclass
class ArtboardContent(JSONMixin):
    horizontalRulerData: RulerData
    verticalRulerData: RulerData
    backgroundColor: Color
    hasBackgroundColor: bool
    includeBackgroundColorInExport: bool
    isFlowHome: bool
    resizesContent: bool
    windingRule: WindingRule
    layout: Optional[LayoutGrid] = None
    grid: Optional[SimpleGrid] = None
    presetDictionary: Optional[Any] = None


@dataclass_json
@dataclass
class Artboard(LayerGroupOptional, ArtboardContent, LayerGroupRequired, SketchObject):
    pass


# Contains metadata about the Sketch file - information about pages and artboards appearing in the file, fonts used,
# the version of Sketch used to save the file etc.
@dataclass_json
@dataclass
class Meta(JSONMixin):
    commit: str
    pagesAndArtboards: Dict[str, MetaPage]
    version: Union[
        Literal[121],
        Literal[122],
        Literal[123],
        Literal[124],
        Literal[125],
        Literal[126],
        Literal[127],
        Literal[128],
        Literal[129],
        Literal[130],
        Literal[131],
        Literal[132],
        Literal[133],
        Literal[134],
        Literal[135],
        Literal[136]
    ]
    compatibilityVersion: Literal[99]
    app: BundleId
    autosaved: NumericalBool
    variant: SketchVariant
    created: MetaCreate
    saveHistory: List[str]
    appVersion: str
    build: int
    coeditCompatibilityVersion: Optional[int] = None


@dataclass_json
@dataclass
class MetaPage(JSONMixin):
    name: str
    artboards: Dict[str, MetaArtboard]


@dataclass_json
@dataclass
class MetaArtboard(JSONMixin):
    name: str


@dataclass_json
@dataclass
class MetaCreate(JSONMixin):
    commit: str
    appVersion: str
    build: int
    app: BundleId
    compatibilityVersion: int
    version: int
    variant: SketchVariant
    coeditCompatibilityVersion: Optional[int] = None


# Enumeration of the Apple bundle ids for the various variants of Sketch
class BundleId(Enum):
    PublicRelease = 'com.bohemiancoding.sketch3'
    Beta = 'com.bohemiancoding.sketch3.beta'
    Private = 'com.bohemiancoding.sketch3.private'
    FeaturePreview = 'com.bohemiancoding.sketch3.feature-preview'
    Internal = 'com.bohemiancoding.sketch3.internal'
    Experimental = 'com.bohemiancoding.sketch3.experimental'
    Testing = 'com.bohemiancoding.sketch3.testing'


# A numerical boolean where 0 is false, and 1 is true.
class NumericalBool(Enum):
    True_ = 0
    False_ = 1


# Enumeration of the Sketch variants
SketchVariant = Union[
    Literal['NONAPPSTORE'],
    Literal['APPSTORE'],
    Literal['BETA'],
    Literal['PRIVATE'],
    Literal['FEATURE_PREVIEW'],
    Literal['INTERNAL'],
    Literal['EXPERIMENTAL'],
    Literal['TESTING'],
    Literal['UNITTEST']
]

User = Dict[str, Any]

# The workspace is a folder in the Sketch file archive that can contain arbitrary JSON files, allowing Sketch and 3rd
# party products and tools to store settings that should travel with the Sketch document. To avoid clashes or
# settings being overridden, select a unique name for your workspace file.
Workspace = Dict[Any, Any]


# This schema describes a representation of an expanded Sketch file, that is, a Sketch file that has been unzipped,
# all of its entries parsed to JSON and merged into a single object. A concrete example of an expanded sketch file is
# the return value of the `fromFile` function in `@sketch-hq/sketch-file`
@dataclass
class Contents:
    document: Document
    meta: Meta
    user: User
    workspace: Optional[Workspace] = None


# The document entry in a Sketch file.
@dataclass_json
@dataclass
class Document(SketchObject):
    do_objectID: Uuid
    assets: AssetCollection
    colorSpace: ColorSpace
    currentPageIndex: int
    foreignLayerStyles: List[ForeignLayerStyle]
    foreignSymbols: List[ForeignSymbol]
    foreignTextStyles: List[ForeignTextStyle]
    layerStyles: SharedStyleContainer
    layerTextStyles: SharedTextStyleContainer
    pages: List[Page]
    foreignSwatches: Optional[List[ForeignSwatch]] = None
    layerSymbols: Optional[SymbolContainer] = None
    sharedSwatches: Optional[SwatchContainer] = None
    fontReferences: Optional[List[FontRef]] = None
    documentState: Optional[DocumentState] = None
    patchInfo: Optional[PatchInfo] = None


# Union of all layers
AnyLayer = Union[SymbolMaster, Group, Oval, Polygon, Rectangle, ShapePath, Star,
                 Triangle, ShapeGroup, Text, SymbolInstance, Slice, Hotspot, Bitmap, Page, Artboard]

# Union of all group layers
AnyGroup = Union[SymbolMaster, Group, ShapeGroup, Page, Artboard]

CLASS_MAP: Dict[str, SketchObject] = {
    "triangle": Triangle,
    "textStyle": TextStyle,
    "text": Text,
    "symbolMaster": SymbolMaster,
    "symbolInstance": SymbolInstance,
    "symbolContainer": SymbolContainer,
    "swatchContainer": SwatchContainer,
    "swatch": Swatch,
    "style": Style,
    "stringAttribute": StringAttribute,
    "star": Star,
    "slice": Slice,
    "simpleGrid": SimpleGrid,
    "sharedTextStyleContainer": SharedTextStyleContainer,
    "sharedStyleContainer": SharedStyleContainer,
    "sharedStyle": SharedStyle,
    "shapePath": ShapePath,
    "shapeGroup": ShapeGroup,
    "shadow": Shadow,
    "rulerData": RulerData,
    "rectangle": Rectangle,
    "rect": Rect,
    "polygon": Polygon,
    "paragraphStyle": ParagraphStyle,
    "page": Page,
    "overrideValue": OverrideValue,
    "oval": Oval,
    "layoutGrid": LayoutGrid,
    "innerShadow": InnerShadow,
    "imageCollection": ImageCollection,
    "group": Group,
    "graphicsContextSettings": GraphicsContextSettings,
    "gradientStop": GradientStop,
    "gradient": Gradient,
    "fontReference": FontRef,
    "fontDescriptor": FontDescriptor,
    "fill": Fill,
    "exportOptions": ExportOptions,
    "exportFormat": ExportFormat,
    "curvePoint": CurvePoint,
    "colorControls": ColorControls,
    "color": Color,
    "borderOptions": BorderOptions,
    "border": Border,
    "blur": Blur,
    "bitmap": Bitmap,
    "attributedString": AttributedString,
    "assetCollection": AssetCollection,
    "artboard": Artboard,
    "MSJSONOriginalDataReference": DataRef,
    "MSJSONFileReference": FileRef,
    "MSImmutablePatchInfo": PatchInfo,
    "MSImmutableOverrideProperty": OverrideProperty,
    "MSImmutableInferredGroupLayout": InferredGroupLayout,
    "MSImmutableHotspotLayer": Hotspot,
    "MSImmutableGradientAsset": GradientAsset,
    "MSImmutableFreeformGroupLayout": FreeformGroupLayout,
    "MSImmutableForeignTextStyle": ForeignTextStyle,
    "MSImmutableForeignSymbol": ForeignSymbol,
    "MSImmutableForeignSwatch": ForeignSwatch,
    "MSImmutableForeignLayerStyle": ForeignLayerStyle,
    "MSImmutableFlowConnection": FlowConnection,
    "MSImmutableColorAsset": ColorAsset
}


@dataclass
class SketchFile:
    filepath: str
    contents: Contents


__all__ = [
    'Uuid', 'AssetCollection', 'ImageCollection', 'ColorAsset', 'Color', 'UnitInterval', 'GradientAsset', 'Gradient',
    'GradientType', 'PointString', 'GradientStop', 'FileRef', 'DataRef', 'Base64Data', 'ColorSpace',
    'ForeignLayerStyle', 'SharedStyle', 'Style', 'Border', 'FillType', 'BorderPosition', 'GraphicsContextSettings',
    'BlendMode', 'BorderOptions', 'LineCapStyle', 'LineJoinStyle', 'Blur', 'BlurType', 'Fill', 'PatternFillType',
    'MarkerType', 'WindingRule', 'TextStyle', 'EncodedAttributes', 'TextVerticalAlignment', 'ParagraphStyle',
    'TextHorizontalAlignment', 'TextTransform', 'UnderlineStyle', 'FontDescriptor', 'FontAttributes', 'Shadow',
    'InnerShadow', 'ColorControls', 'ForeignSymbol', 'SymbolMaster', 'BooleanOperation', 'ExportOptions',
    'ExportFormat', 'ExportFileFormat', 'ExportFormatNamingScheme', 'VisibleScaleType', 'Rect', 'FlowConnection',
    'AnimationType', 'LayerListExpanded', 'ResizeType', 'RulerData', 'LayoutGrid', 'SimpleGrid', 'FreeformGroupLayout',
    'InferredGroupLayout', 'InferredLayoutAxis', 'InferredLayoutAnchor', 'OverrideProperty', 'OverrideName', 'Group',
    'Oval', 'PointsRadiusBehaviour', 'CurvePoint', 'CurveMode', 'Polygon', 'Rectangle', 'ShapePath', 'Star', 'Triangle',
    'ShapeGroup', 'Text', 'AttributedString', 'StringAttribute', 'StringAttributeAttributes', 'LineSpacingBehaviour',
    'TextBehaviour', 'PointListString', 'SymbolInstance', 'OverrideValue', 'Slice', 'Hotspot', 'Bitmap',
    'ForeignTextStyle', 'ForeignSwatch', 'Swatch', 'SharedStyleContainer', 'SharedTextStyleContainer',
    'SymbolContainer', 'SwatchContainer', 'FontRef', 'DocumentState', 'PatchInfo', 'Page', 'Artboard', 'Meta',
    'MetaPage', 'MetaArtboard', 'MetaCreate', 'BundleId', 'NumericalBool', 'SketchVariant', 'User', 'Workspace',
    'Contents', 'Document', 'AnyLayer', 'AnyGroup', 'CLASS_MAP', 'SketchFile'
]

if __name__ == "__main__":
    pass
