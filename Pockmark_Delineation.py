"""
Delineate pockmarks in input DEM using derived Bathymetry Position Index.

"""

import os
from pprint import pprint

from qgis.PyQt.QtCore import QCoreApplication  # , QVariant
from qgis.core import (
    QgsProperty,
    QgsProcessing,
    QgsProcessingUtils,
    QgsProcessingException,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterBand,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRange,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterExpression,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterFileDestination,
    QgsProcessingMultiStepFeedback,
    QgsLayerTreeLayer,
    QgsExpression,
    QgsProcessingLayerPostProcessorInterface,
    QgsRasterLayer,
    QgsVectorLayer,
    QgsGeometry,
    QgsFeature,
    edit,
    QgsStyle,
    QgsColorRampShader,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    QgsRasterBandStats,
)
from qgis import processing
from osgeo import gdal, osr, ogr  # noqa
import codecs
import numpy as np

try:
    pprint({'debug': 'pprint'}, sort_dicts=False)
except AttributeError as e:
    print(e)
    
    def pprint(*args, **kwargs) -> None:
        """Substitute pprint with normal print when QGIS Python console is not open."""
        _ = kwargs.pop('sort_dicts', None)
        print(*args, **kwargs)


def set_nodata_geotiff(path: str, band: int = 1, nodata: float = 0.0, calc_stats: bool = True) -> None:
    """Set NoData value for GeoTIFF file using GDAL."""
    ds = gdal.Open(path, gdal.GA_Update)
    band = ds.GetRasterBand(band)
    band.SetNoDataValue(nodata)
    if calc_stats:
        band.ComputeStatistics(approx_ok=False)
    ds = band = None
    return


def set_SingleBandPseudoColor(rlayer, color_ramp_name: str = 'RdBu', vmin: float = None, vmax: float = None):
    """
    Set SingleBandPseudoColor ramp to raster layer.
    
    """
    rstats = rlayer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
    
    if vmin is None:
        vmin = rstats.minimumValue
    if vmax is None:
        vmax = rstats.minimumValue
    
    color_ramp = QgsStyle().defaultStyle().colorRamp(color_ramp_name)
    shader_ramp = QgsColorRampShader(
        minimumValue=vmin,
        maximumValue=vmax,
        colorRamp=color_ramp,
        type=QgsColorRampShader.Interpolated,
        classificationMode=QgsColorRampShader.Continuous,
    )

    lst = [
        QgsColorRampShader.ColorRampItem(vmin, color_ramp.color1()),
        QgsColorRampShader.ColorRampItem(vmax, color_ramp.color2()),
    ]

    lst = (
        [QgsColorRampShader.ColorRampItem(vmin, color_ramp.color1())]
        + [
            QgsColorRampShader.ColorRampItem(float(val), color_ramp.stops()[i].color)
            for i, val in enumerate(list(np.linspace(vmin, vmax, color_ramp.count())[1:-1]))
        ]
        + [QgsColorRampShader.ColorRampItem(vmax, color_ramp.color2())]
    )

    shader_ramp.setColorRampItemList(lst)

    shader = QgsRasterShader()
    shader.setMinimumValue(vmin)
    shader.setMaximumValue(vmax)
    shader.setRasterShaderFunction(shader_ramp)

    renderer = QgsSingleBandPseudoColorRenderer(
        rlayer.dataProvider(),
        band=1,
        shader=shader
    )
    renderer.setClassificationMin(vmin)
    renderer.setClassificationMax(vmax)

    rlayer.setRenderer(renderer)
    rlayer.triggerRepaint()


class PockmarkDelineation(QgsProcessingAlgorithm):
    """
    Delineate pockmarks (depressions) in input raster using derived Bathymetry Position Index (BPI).
    
    """
    
    # PARAMETER
    INPUT = 'INPUT'
    BAND = 'BAND'
    
    DEM_RELIEF_THRESHOLD = 'DEM_RELIEF_THRESHOLD'
    DEM_MASK_DILATION_RADIUS = 'DEM_MASK_DILATION_RADIUS'
    DEM_MASK_DILATION_NITER = 'DEM_MASK_DILATION_NITER'
    
    BPI_RADII = 'BPI_RADII'
    BPI_THRESHOLD = 'BPI_THRESHOLD'
    BPI_SMOOTH = 'BPI_SMOOTH'
    BPI_SMOOTH_RADIUS = 'BPI_SMOOTH_RADIUS'
    BPI_MASK_OPENING_RADIUS = 'BPI_MASK_OPENING_RADIUS'
    BPI_MASK_OPENING_NITER = 'BPI_MASK_OPENING_NITER'
    BPI_MASK_BUFFER_DISTANCE = 'BPI_MASK_BUFFER_DISTANCE'
    
    SMOOTHING_DIST = 'SMOOTHING_DIST'
    
    OUTPUT = 'OUTPUT'
    OUTPUT_DEM_DEPRESSIONS = 'OUTPUT_DEM_DEPRESSIONS'
    OUTPUT_BPI = 'OUTPUT_BPI'
    OUTPUT_BPI_MASK = 'OUTPUT_BPI_MASK'
    OUTPUT_LOG = 'OUTPUT_LOG'
    
    FILTER_EXPRESSION_BPI = 'FILTER_EXPRESSION_BPI'
    BUFFER_EXPRESSION = 'BUFFER_EXPRESSION'
    
    DEBUG_MODE = 'DEBUG_MODE'
    
    # DEFAULTS
    PARAMETERS = {}
    NODATA = 0
    OUTPUT_FILES = None
        
    # INTERNALS
    RESULTS = {}
    OUTPUTS = {}
    POST_PROCESSORS = {}
    
    
    def tr(self, string):  # noqa
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):  # noqa
        return PockmarkDelineation()

    def name(self):  # noqa
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'pockmarkdelineation'

    def displayName(self):  # noqa
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('(1) Pockmark delineation')

    def group(self):  # noqa
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr('Pockmark analysis')

    def groupId(self):  # noqa
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'pockmarkanalysis'

    def shortHelpString(self):  # noqa
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it..
        """
        return self.tr('Delineate horizon pockmarks')
    
    def printFeedback(self, func, msg: str, prefix=f'\n{80 * "="}\n', suffix=f'\n{80 * "="}'):
        """Wrap feedback.push* functions."""
        func(self.tr(prefix + msg + suffix))
    
    def generateTempFilename(self, filename, context):
        """Generate temporary filename using processing algorithm context."""
        return QgsProcessingUtils.generateTempFilename(filename, context)

    def createHTML(self, output_html, params):
        """
        Create HTML log file.
        
        Source
        ------
            https://github.com/qgis/QGIS/blob/ltr-3_4/python/plugins/processing/algs/qgis/RasterLayerStatistics.py
        
        """
        with codecs.open(output_html, 'w', encoding='utf-8') as f:
            f.write('<html><head>\n')
            f.write('<meta http-equiv="Content-Type" content="text/html; \
                    charset=utf-8" /></head><body>\n')
            
            # Write INPUT filename
            input = params.pop('INPUT')
            f.write('<p><b>{fname}</b></p>\n'.format(fname=str(input)))
            
            # Write parameters
            for key, val in params.items():
                # print(key, val)
                if 'OUTPUT' not in key:
                    f.write('<p>{k}: <b>{v}</b></p>\n'.format(k=str(key), v=str(val)))
            f.write('</body></html>\n')
    
    def initAlgorithm(self, config=None):
        """
        Define the algorithm inputs and outputs of the algorithm as well as additional parameters.
        
        """
        # INPUT
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT,
                description=self.tr('<b>Input DEM</b> (bathymetry, surface, etc)'),
                defaultValue=None,
                optional=False
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                name=self.BAND,
                description=self.tr('Band number'),
                defaultValue=1,
                parentLayerParameterName=self.INPUT,
                optional=False,
                allowMultiple=False
            )
        )
        
        # PARAMETER
        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.DEM_RELIEF_THRESHOLD,
                description=self.tr('<b>[DEM]</b> Depression depth threshold (m) (<i>Fill Sinks</i> algorithm, absolute value)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.1,
                minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterRange(
                name=self.BPI_RADII,
                description=self.tr('<b>[BPI]</b> Radius min/max (grid cells)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=[0, 150],
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BPI_THRESHOLD,
                description=self.tr('<b>[BPI]</b> ' + r'Depression detection threshold (BPI &lt;= threshold)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=-0.5,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BPI_SMOOTH_RADIUS,
                description=self.tr('<b>[BPI]</b> Smoothing radius (grid cells)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=2,
                minValue=0,
            )
        )
        self.parameterDefinition(self.BPI_SMOOTH_RADIUS).setFlags(
            QgsProcessingParameterDefinition.FlagAdvanced
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BPI_MASK_OPENING_RADIUS,
                description=self.tr('<b>[BPI filter]</b> Radius (grid cells) of <i>Morphological Opening</i>'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=0,
            )
        )
        self.parameterDefinition(self.BPI_MASK_OPENING_RADIUS).setFlags(
            QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BPI_MASK_OPENING_NITER,
                description=self.tr('<b>[BPI filter]</b> Iterations (#) of <i>Morphological Opening</i>'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=0,
            )
        )
        self.parameterDefinition(self.BPI_MASK_OPENING_NITER).setFlags(
            QgsProcessingParameterDefinition.FlagAdvanced
        )
        
        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.BPI_MASK_BUFFER_DISTANCE,
                description=self.tr('<b>[BPI]</b> Buffer distance for DEM clipping (m)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=100,  # m
                minValue=0,
            )
        )
        self.parameterDefinition(self.BPI_MASK_BUFFER_DISTANCE).setFlags(
            QgsProcessingParameterDefinition.FlagAdvanced
        )
        
        self.addParameter(
            QgsProcessingParameterExpression(
                name=self.FILTER_EXPRESSION_BPI,
                description=self.tr('<b>[Polygon filter]</b> Exclude polygonized pockmarks via <i>expression</i> (based on <i>BPI</i>, <i>SLOPE</i>, or <i>DEPTH</i>)'),
                defaultValue='"BPI_range" > 0.15',
                optional=True,
            )
        )
        self.parameterDefinition(self.FILTER_EXPRESSION_BPI).setFlags(
            QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(
            QgsProcessingParameterExpression(
                name=self.BUFFER_EXPRESSION,
                description=self.tr('<b>[Polygon buffer]</b> Set buffer distance via <i>expression</i> (m)'),
                defaultValue='round(1/radians(max("SLOPE_mean", 1)) * {cellsize}/2, 1)',
                optional=True,
            )
        )
        self.parameterDefinition(self.BUFFER_EXPRESSION).setFlags(
            QgsProcessingParameterDefinition.FlagAdvanced
        )
        
        self.addParameter(
            QgsProcessingParameterBoolean(
                name=self.DEBUG_MODE,
                description=self.tr('[DEBUG] Load temporary outputs to QGIS?'),
                defaultValue=True,
            )
        )
        self.parameterDefinition(self.DEBUG_MODE).setFlags(
            QgsProcessingParameterDefinition.FlagAdvanced
        )
        
        # OUTPUT
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                name=self.OUTPUT,
                description='<b>Pockmarks (delineated)</b>',
                type=QgsProcessing.TypeVectorPolygon,
                createByDefault=True,
                supportsAppend=True,
                defaultValue=None,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_DEM_DEPRESSIONS,
                description=self.tr('DEM depression mask (Fill algorithm)'),
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_BPI,
                description=self.tr('BPI (calculated)'),
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                name=self.OUTPUT_BPI_MASK,
                description=self.tr('BPI mask (processed)'),
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                name=self.OUTPUT_LOG,
                description=self.tr('Parameter log file (HMTL format)'),
                fileFilter=self.tr('HTML files (*.html)'),
                optional=True,
                createByDefault=True,
            )
        )

    def processAlgorithm(self, parameters, context, model_feedback):
        """
        Delineate pockmarks.
        """
        feedback = QgsProcessingMultiStepFeedback(11, model_feedback)
        self.PARAMETERS = parameters
        
        # ==============================================================================================================
        
        # --------------------------------------------------------------------------------------------------------------
        # INPUT
        # --------------------------------------------------------------------------------------------------------------
        DEM = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        if DEM is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))
        cellsizes = (DEM.rasterUnitsPerPixelX(), DEM.rasterUnitsPerPixelY())
        self.PARAMETERS['CELLSIZES'] = cellsizes
        BAND = self.parameterAsInt(parameters, self.BAND, context)
        self.SOURCE_NAME = DEM.name()
        
        # --------------------------------------------------------------------------------------------------------------
        # PARAMETER
        # --------------------------------------------------------------------------------------------------------------
        self.DEBUG = self.parameterAsBoolean(parameters, self.DEBUG_MODE, context)
        
        DEM_THRESHOLD = self.parameterAsDouble(parameters, self.DEM_RELIEF_THRESHOLD, context)  # noqa
        
        bpi_radii = self.parameterAsRange(parameters, self.BPI_RADII, context)
        bpi_radius_inner, bpi_radius_outer = [int(r) for r in bpi_radii]
        if any(r < 0 for r in bpi_radii):
            raise QgsProcessingException(
                f'BPI radii must be > 0 (min: {bpi_radius_inner} max: {bpi_radius_outer})'
            )
        bpi_threshold = self.parameterAsDouble(parameters, self.BPI_THRESHOLD, context)
        BPI_SMOOTH_RADIUS = self.parameterAsInt(parameters, self.BPI_SMOOTH_RADIUS, context)
        self.BPI_SMOOTHING = True if BPI_SMOOTH_RADIUS > 0 else False
        BPI_MASK_OPENING_RADIUS = self.parameterAsInt(parameters, self.BPI_MASK_OPENING_RADIUS, context)
        BPI_MASK_OPENING_NITER = self.parameterAsInt(parameters, self.BPI_MASK_OPENING_NITER, context)
        BPI_MASK_BUFFER_DISTANCE = self.parameterAsInt(parameters, self.BPI_MASK_BUFFER_DISTANCE, context)
                
        # [EXPRESSION] BPI filter
        bpi_filter_expression = self.parameterAsExpression(parameters, self.FILTER_EXPRESSION_BPI, context)
        BPI_FILTER_EXPRESSION = None if bpi_filter_expression == 'None' else bpi_filter_expression
        
        # [EXPRESSION] Pockmark polygon buffer distance
        buffer_expression = self.parameterAsExpression(parameters, self.BUFFER_EXPRESSION, context)
        if '{cellsize}' in buffer_expression:
            BUFFER_DISTANCE_EXPRESSION = buffer_expression.format(cellsize=max(cellsizes))
            self.PARAMETERS[self.BUFFER_EXPRESSION] = BUFFER_DISTANCE_EXPRESSION
        else:
            BUFFER_DISTANCE_EXPRESSION = buffer_expression
        assert not QgsExpression(BUFFER_DISTANCE_EXPRESSION).hasParserError()
        BUFFER_DISTANCE_EXPRESSION = QgsProperty.fromExpression(BUFFER_DISTANCE_EXPRESSION)

        # Smoothing distance for pockmark polygons (based on grid cellsize)
        SMOOTHING_DISTANCE = max(cellsizes)  # if SMOOTHING_DISTANCE == -1 else SMOOTHING_DISTANCE
        self.PARAMETERS[self.SMOOTHING_DIST] = SMOOTHING_DISTANCE
        
        # --------------------------------------------------------------------------------------------------------------
        # OUTPUTS
        # --------------------------------------------------------------------------------------------------------------
        OUTPUT_HTML = self.parameterAsFileOutput(parameters, self.OUTPUT_LOG, context)
        
        if self.DEBUG:
            print('\n' + '=' * 80)
            pprint(parameters, sort_dicts=False)
            print('[BPI_FILTER_EXPRESSION]  ', bpi_filter_expression, type(bpi_filter_expression))
            print('[BPI_FILTER_EXPRESSION]  ', BPI_FILTER_EXPRESSION)
            print('[BUFFER_DISTANCE_EXPRESSION]  ', buffer_expression, type(buffer_expression))
            print('[BUFFER_DISTANCE_EXPRESSION]  ', BUFFER_DISTANCE_EXPRESSION)
                
        # ==============================================================================================================
        # [BPI]
        # ==============================================================================================================
        self.printFeedback(feedback.pushInfo, '[BPI] Calculate BPI (r_inner={rin}, r_outer={rout})'.format(rin=bpi_radius_inner, rout=bpi_radius_outer))
        self.OUTPUTS['BPI'] = processing.run('sagang:topographicpositionindextpi', {
            'DEM': DEM,
            'STANDARD': True,
            'RADIUS_MIN': bpi_radius_inner,
            'RADIUS_MAX': bpi_radius_outer,
            'DW_WEIGHTING': 0,  # no weighting
            'TPI': self.generateTempFilename('BPI.tif', context=context) if self.BPI_SMOOTHING else parameters[self.OUTPUT_BPI],
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['TPI']
        
        if feedback.isCanceled():
            return {}
        
        if self.BPI_SMOOTHING:
            self.printFeedback(feedback.pushInfo, '[BPI] Smooth BPI grid (radius={r})'.format(r=BPI_SMOOTH_RADIUS))
            self.OUTPUTS['BPI_smoothed'] = processing.run('sagang:simplefilter', {
                'INPUT': self.OUTPUTS['BPI'],
                'METHOD': 0,  # Smooth
                'KERNEL_TYPE': 1,  # Circle
                'KERNEL_RADIUS': BPI_SMOOTH_RADIUS,
                'RESULT': parameters[self.OUTPUT_BPI],
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['RESULT']
            BPI_GRID = self.OUTPUTS['BPI_smoothed']
        else:
            BPI_GRID = self.OUTPUTS['BPI']
            
        if feedback.isCanceled():
            return {}
        
        # Clip BPI raster using user-defined threshold
        self.printFeedback(feedback.pushInfo, '[BPI] Mask BPI raster using threshold (BPI <= {t})'.format(t=bpi_threshold))
        FORMULA = '(A <= {cutoff})'.format(cutoff=bpi_threshold)
        self.OUTPUTS['BPI_mask_threshold'] = processing.run('gdal:rastercalculator', {
            'INPUT_A': BPI_GRID,
            'BAND_A': BAND,
            'FORMULA': FORMULA,
            'NO_DATA': 0.0,
            'RTYPE': 5,  # 5: Float32
            'OUTPUT': self.generateTempFilename('BPI_mask_threshold.tif', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        # [Morphological Opening] Remove small isolated patches
        self.printFeedback(feedback.pushInfo, '[BPI] Remove small, isolated mask patches via morphological opening')
        self.OUTPUTS['BPI_mask_morph_opening'] = processing.run('sagang:morphologicalfilteropencv', {
            'INPUT': self.OUTPUTS['BPI_mask_threshold'],
            # 'INPUT': self.OUTPUTS['BPI_mask_morph_fill_holes'],
            'TYPE': 2,   # 2: opening
            'SHAPE': 0,  # 0: ellipse, 1: rectangle
            'RADIUS': BPI_MASK_OPENING_RADIUS,
            'ITERATIONS': BPI_MASK_OPENING_NITER,
            'OUTPUT': parameters[self.OUTPUT_BPI_MASK],
            # 'OUTPUT': self.generateTempFilename('BPI_mask_morph_opening.tif', context=context),
            # 'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        set_nodata_geotiff(self.OUTPUTS['BPI_mask_morph_opening'], band=BAND, nodata=self.NODATA)
        
        if feedback.isCanceled():
            return {}
        
        # Polygonize BPI mask
        self.printFeedback(feedback.pushInfo, '[BPI] Polygonize BPI depressions')
        self.OUTPUTS['polygons_BPI'] = processing.run('gdal:polygonize', {
            'INPUT': self.OUTPUTS['BPI_mask_morph_opening'],
            'BAND': BAND,
            'FIELD': 'DN',
            'EIGHT_CONNECTEDNESS': False,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            # 'OUTPUT': self.generateTempFilename('polygons.tif', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        RADIUS = BPI_MASK_BUFFER_DISTANCE / max(cellsizes)
        self.OUTPUTS['BPI_mask_buffer'] = processing.run('sagang:morphologicalfilteropencv', {
            'INPUT': self.OUTPUTS['BPI_mask_morph_opening'],
            'TYPE': 0,   # 0: dilation
            'SHAPE': 0,  # 0: ellipse
            'RADIUS': RADIUS,
            'ITERATIONS': 1,
            'OUTPUT': self.generateTempFilename('BPI_mask_buffer.tif', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        # ==============================================================================================================
        # [DEM]
        # ==============================================================================================================
        self.printFeedback(feedback.pushInfo, '[DEM] Clip DEM with buffered BPI mask')
        self.OUTPUTS['DEM_clipped_bpi'] = processing.run('gdal:rastercalculator', {
            'INPUT_A': DEM,
            'BAND_A': BAND,
            'INPUT_B': self.OUTPUTS['BPI_mask_buffer'],
            'BAND_B': 1,
            'FORMULA': 'A * (B == 1)',
            'NO_DATA': 0.0,
            'RTYPE': 5,  # 5: Float32
            'OUTPUT': self.generateTempFilename('DEM_clipped_bpi.tif', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.printFeedback(feedback.pushInfo, '[DEM] Buffer (extrapolate) clipped DEM (2 pixel)')
        self.OUTPUTS['DEM_clipped_bpi_buffer'] = processing.run('gdal:fillnodata', {
            'INPUT': self.OUTPUTS['DEM_clipped_bpi'],
            'BAND': BAND,
            'DISTANCE': 2,  # Maximum distance (in pixels) to search out for values to interpolate
            'ITERATIONS': 0,  # Smoothing iterations
            'OUTPUT': self.generateTempFilename('DEM_clipped_bpi_buffer.tif', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.printFeedback(feedback.pushInfo, '[DEM] Create border for boundary features')
        DEM_NoDataValue = DEM.dataProvider().sourceNoDataValue(BAND)
        print(f'{DEM_NoDataValue = }')
        self.OUTPUTS['DEM_fenced'] = processing.run('gdal:rastercalculator', {
            'INPUT_A': DEM,
            'BAND_A': BAND,
            'INPUT_B': self.OUTPUTS['DEM_clipped_bpi_buffer'],
            'BAND_B': 1,
            'INPUT_C': self.OUTPUTS['DEM_clipped_bpi'],
            'BAND_C': 1,
            'FORMULA': 'where(logical_and(A <= {nan}, B != 0), 1, C)'.format(nan=DEM_NoDataValue),
            'NO_DATA': 0.0,
            'RTYPE': 5,  # 5: Float32
            'EXTRA': '--hideNoData',
            'OUTPUT': self.generateTempFilename('DEM_fenced.tif', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.printFeedback(feedback.pushInfo, '[DEM] Fill depressiosn')
        self.OUTPUTS['DEM_filled'] = processing.run('sagang:fillsinkswangliu', {
            'ELEV': self.OUTPUTS['DEM_fenced'],
            'MINSLOPE': 0.01,
            'FDIR': 'none',
            'WSHED': 'none',
            'FILLED': self.generateTempFilename('DEM_filled.tif', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['FILLED']
        # set_nodata_geotiff(self.OUTPUTS['DEM_filled'], band=BAND, nodata=self.NODATA)
        
        if feedback.isCanceled():
            return {}
        
        self.printFeedback(feedback.pushInfo, '[DEM] Create depression mask')
        FORMULA_BPI = 'abs(A - B) > {t}'.format(t=DEM_THRESHOLD)
        self.OUTPUTS['DEM_depressions_mask'] = processing.run('gdal:rastercalculator', {
            'INPUT_A': DEM,
            'BAND_A': BAND,
            'INPUT_B': self.OUTPUTS['DEM_filled'],
            'BAND_B': BAND,
            'INPUT_C': self.OUTPUTS['DEM_clipped_bpi'],
            'BAND_C': BAND,
            'FORMULA': FORMULA_BPI,
            'NO_DATA': self.NODATA,
            'RTYPE': 5,  # 5: Float32
            'OUTPUT': self.generateTempFilename('DEM_depressions_mask.tif', context=context),
            # 'OUTPUT': parameters[self.OUTPUT_DEM_DEPRESSIONS],
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        set_nodata_geotiff(self.OUTPUTS['DEM_depressions_mask'], band=BAND, nodata=self.NODATA)
        
        if feedback.isCanceled():
            return {}
        
        self.printFeedback(feedback.pushInfo, '[DEM] Buffer depression mask (infill holes)')
        self.OUTPUTS['DEM_depressions_mask_buffered'] = processing.run('sagang:morphologicalfilteropencv', {
            'INPUT': self.OUTPUTS['DEM_depressions_mask'],
            'TYPE': 0,   # dilation
            'SHAPE': 2,  # cross
            'RADIUS': 1,
            'ITERATIONS': 2,
            # 'OUTPUT': parameters[self.OUTPUT_DEM_DEPRESSIONS],
            'OUTPUT': self.generateTempFilename('DEM_depressions_mask_buffered.tif', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        set_nodata_geotiff(self.OUTPUTS['DEM_depressions_mask_buffered'], band=BAND, nodata=self.NODATA)
        
        if feedback.isCanceled():
            return {}
        
        # ==============================================================================================================
        # POCKMARK POLYGONS
        # ==============================================================================================================
        
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Create polygon raster mask')
        self.OUTPUTS['depressions_mask'] = processing.run('gdal:rastercalculator', {
            'INPUT_A': BPI_GRID,
            'BAND_A': 1,
            'INPUT_B': self.OUTPUTS['BPI_mask_buffer'],
            'BAND_B': 1,
            'INPUT_C': self.OUTPUTS['DEM_depressions_mask_buffered'],
            'BAND_C': 1,
            'FORMULA': '(A < 0) * (B == 1) * (C == 1)',
            'NO_DATA': self.NODATA,
            'RTYPE': 5,  # 5: Float32
            # 'OUTPUT': self.generateTempFilename('depressions_mask.tif', context=context),
            'OUTPUT': parameters[self.OUTPUT_DEM_DEPRESSIONS],
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        set_nodata_geotiff(self.OUTPUTS['depressions_mask'], band=BAND, nodata=self.NODATA)
        
        if feedback.isCanceled():
            return {}
        
        # Polygonize depression mask
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Polygonize (pockmark) depression mask')
        self.OUTPUTS['polygons'] = processing.run('gdal:polygonize', {
            'INPUT': self.OUTPUTS['depressions_mask'],
            'BAND': BAND,
            'FIELD': 'DN',
            'EIGHT_CONNECTEDNESS': False,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            # 'OUTPUT': self.generateTempFilename('polygons.tif', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        # Calculate Zonal Statistics
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Calculate Zonal Statistics')
        STATISTICS = [2, 3, 4, 5, 6, 7]  # Mean, Median, StdDev, Minimum, Maximum, Range
        
        # (a) on BPI
        self.OUTPUTS['zonal_stats'] = processing.run('native:zonalstatistics', {
            'INPUT_RASTER': BPI_GRID,
            'RASTER_BAND': BAND,
            'INPUT_VECTOR': self.OUTPUTS['polygons'],
            'COLUMN_PREFIX': 'BPI_',
            'STATISTICS': STATISTICS,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['INPUT_VECTOR']
        
        # (b) on SLOPE
        self.OUTPUTS['src_slope'] = processing.run('native:slope', {
            'INPUT': DEM,
            'Z_FACTOR': 1,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        self.OUTPUTS['zonal_stats'] = processing.run('native:zonalstatistics', {
            'INPUT_RASTER': self.OUTPUTS['src_slope'],
            'RASTER_BAND': BAND,
            'INPUT_VECTOR': self.OUTPUTS['zonal_stats'],
            'COLUMN_PREFIX': 'SLOPE_',
            'STATISTICS': STATISTICS,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['INPUT_VECTOR']
        
        # (c) on DEM
        self.OUTPUTS['zonal_stats'] = processing.run('native:zonalstatistics', {
            'INPUT_RASTER': DEM,
            'RASTER_BAND': BAND,
            'INPUT_VECTOR': self.OUTPUTS['zonal_stats'],
            'COLUMN_PREFIX': 'DEPTH_',
            'STATISTICS': STATISTICS,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['INPUT_VECTOR']
        
        if feedback.isCanceled():
            return {}
        
        if BPI_FILTER_EXPRESSION is not None:
            self.printFeedback(feedback.pushInfo, '[POLYGONS] Filter polygons using BPI ({})'.format(BPI_FILTER_EXPRESSION))

            polygons_BPI_filter = processing.run('native:extractbyexpression', {
                'INPUT': self.OUTPUTS['zonal_stats'],
                'EXPRESSION': BPI_FILTER_EXPRESSION,  # '"BPI_range" > 1'
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
                'FAIL_OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
                # 'OUTPUT': self.generateTempFilename('polygons_BPI_filtered.gpkg', context=context),
                # 'FAIL_OUTPUT': self.generateTempFilename('polygons_BPI_filtered-excluded.gpkg', context=context),
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )
            self.OUTPUTS['polygons_filtered'] = polygons_BPI_filter['OUTPUT']
            lyr_filtered_by_bpi = QgsProcessingUtils.mapLayerFromString(polygons_BPI_filter['FAIL_OUTPUT'], context=context)
            features_filtererd_by_BPI = lyr_filtered_by_bpi.featureCount()
            feedback.pushInfo('[VECTOR] Filtered < {n} > features'.format(n=features_filtererd_by_BPI))
            
            if feedback.isCanceled():
                return {}
        
        # Buffer polygons using custom expression
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Buffer polygons using expression ({})'.format(BUFFER_DISTANCE_EXPRESSION))
        self.OUTPUTS['polygons_buffered'] = processing.run('native:buffer', {
            'INPUT': self.OUTPUTS['polygons_filtered'] if BPI_FILTER_EXPRESSION is not None else self.OUTPUTS['zonal_stats'],
            'DISTANCE': BUFFER_DISTANCE_EXPRESSION,
            'SEGMENTS': 8,
            'END_CAP_STYLE': 0,
            'JOIN_STYLE': 0,  # 0: Round, 1: Miter, 2: Bevel
            'DISSOLVE': True,
            'SEPARATE_DISJOINT': True,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            # 'OUTPUT': self.generateTempFilename('polygons_buffered.gpkg', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
            
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Rasterize buffered polygons')
        # Remove `fid` field --> causes issues writing GeoPackage
        lyr_polygons_buffered = QgsProcessingUtils.mapLayerFromString(self.OUTPUTS['polygons_buffered'], context=context)
        fields = lyr_polygons_buffered.fields()
        field_names = [f.name() for f in fields]
        idx = lyr_polygons_buffered.fields().indexFromName('fid')
        if idx != -1:
            with edit(lyr_polygons_buffered):
                check = lyr_polygons_buffered.dataProvider().deleteAttributes([idx])
                lyr_polygons_buffered.updateFields()
            
            if self.DEBUG:
                print('[lyr_vertices] Removed `fid` field`?   ', check)
        
        self.OUTPUTS['raster_polygons_buffered'] = processing.run('gdal:rasterize', {
            'INPUT': lyr_polygons_buffered,
            'BURN': '1',
            'UNITS': 1,  # 0: Pixels, 1: Georeferenced units
            'WIDTH': cellsizes[0],
            'HEIGHT': cellsizes[1],
            'EXTENT': DEM,
            'NODATA': self.NODATA,
            'DATA_TYPE': 1,  # Int16
            'OUTPUT': self.generateTempFilename('raster_polygons_buffered.tif', context=context)
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
            
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Clip buffered polygons to DEM depression mask')
        self.OUTPUTS['raster_polygons_buffered_clip'] = processing.run('gdal:rastercalculator', {
            'INPUT_A': self.OUTPUTS['DEM_depressions_mask_buffered'],
            'BAND_A': 1,
            'INPUT_B': self.OUTPUTS['raster_polygons_buffered'],
            'BAND_B': 1,
            'FORMULA': 'A * B',
            'NO_DATA': self.NODATA,
            'RTYPE': 1,  # Int16
            'OUTPUT': self.generateTempFilename('raster_polygons_buffered_clip.tif', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        # set_nodata_geotiff(self.OUTPUTS['depressions_mask'], band=BAND, nodata=self.NODATA)
        
        if feedback.isCanceled():
            return {}
        
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Polygonize buffered & clipped polygons')
        self.OUTPUTS['polygons_buffered_clip'] = processing.run('gdal:polygonize', {
            'INPUT': self.OUTPUTS['raster_polygons_buffered_clip'],
            'BAND': 1,
            'FIELD': 'DN',
            'EIGHT_CONNECTEDNESS': False,
            'OUTPUT': self.generateTempFilename('polygons_buffered_clip.gpkg', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Extract valid polygons (BPI depression intersection)')
        self.OUTPUTS['polygons_buffered_clip_valid'] = processing.run('native:extractbylocation', {
            'INPUT': self.OUTPUTS['polygons_buffered_clip'],
            'PREDICATE': 0,  # 0: intersect
            'INTERSECT': self.OUTPUTS['polygons_BPI'],
            'OUTPUT': self.generateTempFilename('polygons_buffered_clip_valid.gpkg', context=context),
            # 'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        # --------------------------------------------------------------------------------------------------------------
        # NON-OVERLAPPING BUFFER
        # --------------------------------------------------------------------------------------------------------------
        
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Create non-overlapping buffer mask')
        self.OUTPUTS['polygons_smoothed'] = processing.run('native:smoothgeometry', {
            'INPUT': self.OUTPUTS['polygons_filtered'] if BPI_FILTER_EXPRESSION is not None else self.OUTPUTS['zonal_stats'],
            'ITERATIONS': 1,
            'OFFSET': 0.5,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            # 'OUTPUT': self.generateTempFilename('polygons_smoothed.gpkg', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.OUTPUTS['polygons_smoothed_id'] = processing.run('native:adduniquevalueindexfield', {
            'INPUT': self.OUTPUTS['polygons_smoothed'],
            'FIELD': 'fid',
            'FIELD_NAME': 'id',
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Smooth pockmark polygons')
        self.OUTPUTS['polygons_vertices'] = processing.run('native:extractvertices', {
            'INPUT': self.OUTPUTS['polygons_smoothed_id'],
            # 'OUTPUT': self.generateTempFilename('polygons_vertices.gpkg', context=context),
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT  # REQUIRED!
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        # --------------------------------------------------------------------------------------------------------------
        
        # map vertices as QgsVectorLayer
        lyr_vertices = QgsProcessingUtils.mapLayerFromString(self.OUTPUTS['polygons_vertices'], context=context)
        if self.DEBUG:
            print('[lyr_vertices] CRS:', lyr_vertices.crs())
        lyr_vertices.setCrs(DEM.crs())
        if self.DEBUG:
            print('[lyr_vertices] CRS:', lyr_vertices.crs())
        
        # Remove `fid` field --> causes issues writing GeoPackage
        fields = lyr_vertices.fields()
        field_names = [f.name() for f in fields]
        idx = lyr_vertices.fields().indexFromName('fid')
        if idx != -1:
            with edit(lyr_vertices):
                check = lyr_vertices.dataProvider().deleteAttributes([idx])
                lyr_vertices.updateFields()
            
            if self.DEBUG:
                print('[lyr_vertices] Removed `fid` field`?   ', check)
        
        # get buffered geometry
        geom_extent = QgsGeometry.fromWkt(lyr_vertices.extent().asWktPolygon())
        geom_buffer = geom_extent.buffer(
            500, 1, QgsGeometry.EndCapStyle.Square, QgsGeometry.JoinStyle.Miter, 2.0
        )
        # print(geom_buffer.asWkt())

        lyr_tmp = QgsVectorLayer(f"Point?crs={DEM.crs().authid()}", "tmp_vertices", "memory")
        pr = lyr_tmp.dataProvider()
        if self.DEBUG:
            print('[lyr_tmp] CRS:', lyr_tmp.crs())

        # add raster extent vertices
        for i, geom_pt in enumerate(geom_buffer.vertices()):
            # print(i, geom_pt)
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPoint(geom_pt))
            check = pr.addFeature(f)
            if not check:
                print('Could not load vertices!')
        
        if self.DEBUG:
            print('[lyr_tmp] CRS:', lyr_tmp.crs())
        
        self.OUTPUTS['polygons_vertices_append'] = processing.run('native:mergevectorlayers', {
            'LAYERS': [lyr_vertices, lyr_tmp],
            'CRS': DEM.crs(),
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            # 'OUTPUT': self.generateTempFilename('polygons_vertices_append.gpkg', context=context),  # no projection assigned...
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        lyr_vertices_append = QgsProcessingUtils.mapLayerFromString(self.OUTPUTS['polygons_vertices_append'], context=context)
        if self.DEBUG:
            print('[lyr_vertices_append] CRS:', lyr_vertices_append.crs())
        lyr_vertices_append.setCrs(DEM.crs())
        if self.DEBUG:
            print('[lyr_vertices_append] CRS:', lyr_vertices_append.crs())
        
        if feedback.isCanceled():
            return {}
        
        # --------------------------------------------------------------------------------------------------------------
        
        self.OUTPUTS['voronoi_polygons'] = processing.run('native:voronoipolygons', {
            'INPUT': lyr_vertices_append,  # self.OUTPUTS['polygons_vertices_append'],
            'BUFFER': 0,
            'TOLERANCE': 0,
            'COPY_ATTRIBUTES': True,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            # 'OUTPUT': self.generateTempFilename('voronoi_polygons.gpkg', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.OUTPUTS['voronoi_polygons_dissolved'] = processing.run('native:dissolve', {
            'INPUT': self.OUTPUTS['voronoi_polygons'],
            'FIELD': 'id',
            'SEPARATE_DISJOINT': True,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            # 'OUTPUT': self.generateTempFilename('voronoi_polygons_dissolved.gpkg', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.OUTPUTS['polygons_buffered_clipped'] = processing.run('native:intersection', {
            'INPUT': self.OUTPUTS['polygons_buffered_clip_valid'],  # !!! 'polygons_buffered'
            'OVERLAY': self.OUTPUTS['voronoi_polygons_dissolved'],
            'INPUT_FIELDS': 'DN',
            'OVERLAY_FIELDS': '',
            'OVERLAY_FIELDS_PREFIX': '',
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            # 'OUTPUT': self.generateTempFilename('polygons_buffered_clipped.gpkg', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        EXPRESSION_REMOVE_DUPLICATE_FID = '$area = maximum($area, group_by:="id")'
        self.OUTPUTS['polygons_buffered_clipped_filtered'] = processing.run('native:extractbyexpression', {
            'INPUT': self.OUTPUTS['polygons_buffered_clipped'],
            'EXPRESSION': EXPRESSION_REMOVE_DUPLICATE_FID,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            # 'OUTPUT': self.generateTempFilename('polygons_buffered_clipped_filtered.gpkg', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
         
        # --------------------------------------------------------------------------------------------------------------
        # POCKMARK POLYGONS (FINAL)
        # --------------------------------------------------------------------------------------------------------------

        # self.printFeedback(feedback.pushInfo, '[POLYGONS] Clip polygonized DEM depression with BPI pockmark polygons')
        # self.OUTPUTS['polygons_pockmarks'] = processing.run('native:intersection', {
        #     'INPUT': self.OUTPUTS['polygons_buffered_clipped_filtered'],
        #     'OVERLAY': self.OUTPUTS['DEM_depressions_polygonized_noHoles'],
        #     'INPUT_FIELDS': '',
        #     'OVERLAY_FIELDS': 'DN',
        #     'OVERLAY_FIELDS_PREFIX': '',
        #     'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
        #     # 'OUTPUT': self.generateTempFilename('polygons_pockmarks.gpkg', context=context),
        #     },
        #     is_child_algorithm=True,
        #     context=context,
        #     feedback=feedback,
        # )['OUTPUT']
        
        # if feedback.isCanceled():
        #     return {}
        
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Simplify & smooth pockmark polygons')
        self.OUTPUTS['_polygons_pockmarks_buffer_inv'] = processing.run('native:buffer', {
            'INPUT': self.OUTPUTS['polygons_buffered_clipped_filtered'],
            'DISTANCE': -SMOOTHING_DISTANCE,
            'SEGMENTS': 8,
            'END_CAP_STYLE': 0,
            'JOIN_STYLE': 0,  # 0: Round, 1: Miter, 2: Bevel
            'DISSOLVE': False,
            'SEPARATE_DISJOINT': True,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
            
        # EXPRESSION_REMOVE_DUPLICATE_FID = '$area = maximum($area, group_by:="id")'
        self.OUTPUTS['_polygons_pockmarks_buffer_inv_filt'] = processing.run('native:extractbyexpression', {
            'INPUT': self.OUTPUTS['_polygons_pockmarks_buffer_inv'],
            'EXPRESSION': EXPRESSION_REMOVE_DUPLICATE_FID,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.OUTPUTS['_polygons_pockmarks_simple'] = processing.run('native:simplifygeometries', {
            'INPUT': self.OUTPUTS['_polygons_pockmarks_buffer_inv_filt'],
            'METHOD': 0,  # Distance (Douglas-Peucker)
            'TOLERANCE': SMOOTHING_DISTANCE,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.OUTPUTS['_polygons_pockmarks_buffer'] = processing.run('native:buffer', {
            'INPUT': self.OUTPUTS['_polygons_pockmarks_simple'],
            'DISTANCE': SMOOTHING_DISTANCE,
            'SEGMENTS': 8,
            'END_CAP_STYLE': 0,
            'JOIN_STYLE': 0,  # 0: Round, 1: Miter, 2: Bevel
            'DISSOLVE': False,
            'SEPARATE_DISJOINT': True,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.OUTPUTS['_polygons_pockmarks_simple2'] = processing.run('native:simplifygeometries', {
            'INPUT': self.OUTPUTS['_polygons_pockmarks_buffer'],
            'METHOD': 0,  # Distance (Douglas-Peucker)
            'TOLERANCE': SMOOTHING_DISTANCE,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        EXPRESSION_GEOM = 'not is_empty_or_null($geometry)'
        self.OUTPUTS['_polygons_pockmarks_geom'] = processing.run('native:extractbyexpression', {
            'INPUT': self.OUTPUTS['_polygons_pockmarks_simple2'],
            'EXPRESSION': EXPRESSION_GEOM,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
            
        self.OUTPUTS['polygons_pockmarks_output'] = processing.run('native:smoothgeometry', {
            'INPUT': self.OUTPUTS['_polygons_pockmarks_geom'],
            'ITERATIONS': 3,
            'OFFSET': 0.25,
            'MAX_ANGLE': 180,
            'OUTPUT': self.generateTempFilename('polygons_pockmarks_output.gpkg', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        # --------------------------------------------------------------------------------------------------------------
        # MINIMUM BOUNDING GEOMETRY
        # --------------------------------------------------------------------------------------------------------------
        
        self.printFeedback(feedback.pushInfo, '[MBG] Compute Minimum Bounding Geometry (rectangle by width)')
        self.OUTPUTS['MBG_polygons_pockmarks'] = processing.run('wbt:MinimumBoundingBox', {
            'input': self.OUTPUTS['polygons_pockmarks_output'],
            'criterion': 2,  # width
            'features': True,  # Find bounding rectangles around each individual feature
            'output': self.generateTempFilename('MBG_polygons_pockmarks.shp', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['output']
        
        prjfile = self.OUTPUTS['MBG_polygons_pockmarks'].replace('.shp', '.prj')
        if not os.path.isfile(prjfile):
            feedback.pushWarning('Creating missing ESRI projection file (*.prj)')
            with open(prjfile, 'w') as f:
                spatialRef = osr.SpatialReference()
                spatialRef.ImportFromEPSG(DEM.crs().postgisSrid())
                spatialRef.MorphToESRI()
                f.write(spatialRef.ExportToWkt())
        
        self.OUTPUTS['MBG_polygons_pockmarks_stats'] = processing.run('qgis:minimumboundinggeometry', {
            'INPUT': self.OUTPUTS['MBG_polygons_pockmarks'],
            'FIELD': 'fid',
            'TYPE': 1,  # Minimum Oriented Rectangle
            'OUTPUT': self.generateTempFilename('MBG_polygons_pockmarks.gpkg', context=context),
            # 'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        # --------------------------------------------------------------------------------------------------------------
        # STATISTICS
        # --------------------------------------------------------------------------------------------------------------
        
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Spatial join MBG stats with pockmark polygons')
        polygons_stats = processing.run('native:joinattributestable', {
            'INPUT': self.OUTPUTS['polygons_pockmarks_output'],
            'FIELD': 'fid',
            'INPUT_2': self.OUTPUTS['MBG_polygons_pockmarks_stats'],
            'FIELD_2': 'fid',
            'FIELDS_TO_COPY': ['width', 'height', 'angle'],
            'PREFIX': 'MBG_',
            'METHOD': 1,  # Take attributes of the first matching feature only (one-to-one)
            'DISCARD_NONMATCHING': True,
            # 'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            'NON_MATCHING': QgsProcessing.TEMPORARY_OUTPUT,
            'OUTPUT': self.generateTempFilename('polygons_pockmarks_stats.gpkg', context=context),
            # 'NON_MATCHING': self.generateTempFilename('polygons_pockmarks_stats_failed.gpkg', context=context)
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )
        self.OUTPUTS['polygons_pockmarks_stats'] = polygons_stats['OUTPUT']
        self.OUTPUTS['polygons_pockmarks_stats_failed'] = polygons_stats['NON_MATCHING']
        
        feedback.pushInfo(self.tr('JOINED_COUNT: ' + str(polygons_stats['JOINED_COUNT'])))
        feedback.pushWarning(self.tr('UNJOINABLE_COUNT: ' + str(polygons_stats['UNJOINABLE_COUNT'])))
        
        if feedback.isCanceled():
            return {}
        
        # [HOUSEKEEPING} Remove attribute fields
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Remove unused attribute fields')
        lyr_stats = QgsProcessingUtils.mapLayerFromString(
            self.OUTPUTS['polygons_pockmarks_stats'], context=context
        )
        dataprovider_stats = lyr_stats.dataProvider()
        if self.DEBUG:
            print('[lyr_stats] CRS:', lyr_stats.crs())
        lyr_stats.setCrs(DEM.crs())
        if self.DEBUG:
            print('[lyr_stats] CRS:', lyr_stats.crs())
        
        fields = lyr_stats.fields()
        field_names = [field.name() for field in fields]
        field_names_keep = [
            f for f in field_names if any(c in f for c in ['fid', 'MBG_'])  # 'BPI_', 'SLOPE_', 'DEPTH_',
        ]
        field_names_delete = list(set(field_names) - set(field_names_keep))
        field_names_delete_idx = [fields.indexFromName(fname) for fname in field_names_delete]
        with edit(lyr_stats):
            check = dataprovider_stats.deleteAttributes(field_names_delete_idx)
            dataprovider_stats.renameAttributes({
                dataprovider_stats.fieldNameIndex('MBG_width'): 'MBG_shortAxis',
                dataprovider_stats.fieldNameIndex('MBG_height'): 'MBG_longAxis',
            })
        
        lyr_stats.updateFields()
        self.OUTPUTS['lyr_stats'] = lyr_stats
        if self.DEBUG:
            print('[lyr_stats] CRS:', lyr_stats.crs())
        
        # Calcultate additional fields
        tmp_1 = processing.run('native:fieldcalculator', {
            'INPUT': lyr_stats,
            'FIELD_NAME': 'MBG_aspect_ratio',
            'FORMULA': '"MBG_shortAxis" / "MBG_longAxis"',
            'FIELD_TYPE': 0,  # decimal (double)
            'FIELD_LENGTH': 10,
            'FIELD_PRECISION': 3,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Update zonal statistics')
        # (a) on BPI
        zonalstatistics_bpi = processing.run('native:zonalstatistics', {
            'INPUT_RASTER': BPI_GRID,
            'RASTER_BAND': BAND,
            'INPUT_VECTOR': tmp_1,
            'COLUMN_PREFIX': 'BPI_',
            'STATISTICS': STATISTICS,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['INPUT_VECTOR']
        
        # (b) on SLOPE
        zonalstatistics_slope = processing.run('native:zonalstatistics', {
            'INPUT_RASTER': self.OUTPUTS['src_slope'],
            'RASTER_BAND': BAND,
            'INPUT_VECTOR': zonalstatistics_bpi,
            'COLUMN_PREFIX': 'SLOPE_',
            'STATISTICS': STATISTICS,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['INPUT_VECTOR']
        
        # (c) on DEM
        zonalstatistics_dem = processing.run('native:zonalstatistics', {  # noqa
            'INPUT_RASTER': DEM,
            'RASTER_BAND': BAND,
            'INPUT_VECTOR': zonalstatistics_slope,
            'COLUMN_PREFIX': 'DEPTH_',
            'STATISTICS': STATISTICS,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['INPUT_VECTOR']
        
        if feedback.isCanceled():
            return {}
        
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Calculate additional attribute fields (MBG_W_H, area, perimeter)')
        tmp_2 = processing.run('native:fieldcalculator', {
            'INPUT': tmp_1,
            'FIELD_NAME': 'GEOM_area',
            'FORMULA': '$area',
            'FIELD_TYPE': 0,  # decimal (double)
            'FIELD_LENGTH': 10,
            'FIELD_PRECISION': 3,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        self.OUTPUTS['polygons_pockmarks_stats_out'] = processing.run('native:fieldcalculator', {
            'INPUT': tmp_2,
            'FIELD_NAME': 'GEOM_perimeter',
            'FORMULA': '$perimeter',
            'FIELD_TYPE': 0,  # decimal (double)
            'FIELD_LENGTH': 10,
            'FIELD_PRECISION': 3,
            'OUTPUT': parameters[self.OUTPUT],
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        if feedback.isCanceled():
            return {}
        
        # ==============================================================================================================
        
        self.printFeedback(feedback.pushInfo, '', suffix='')
        
        # set RESULTS
        self.RESULTS[self.OUTPUT_DEM_DEPRESSIONS] = self.OUTPUTS['depressions_mask']
        self.RESULTS[self.OUTPUT_BPI] = BPI_GRID
        self.RESULTS[self.OUTPUT_BPI_MASK] = self.OUTPUTS['BPI_mask_morph_opening']
        self.RESULTS[self.OUTPUT] = self.OUTPUTS['polygons_pockmarks_stats_out']
        
        self.createHTML(OUTPUT_HTML, self.PARAMETERS)
        self.RESULTS[self.OUTPUT_LOG] = OUTPUT_HTML
        
        # set reference for postProcessingAlgorithm()
        self.GROUP_SUFFIX = bpi_radii
        
        if self.DEBUG:
            print('\n' + '-' * 80)
            print('OUTPUTS')
            pprint(self.OUTPUTS, sort_dicts=False)
            print('\n' + '-' * 80)
            print('RESULTS')
            pprint(self.RESULTS, sort_dicts=False)
        
        return self.RESULTS
    
    def load_temporary_layer(
        self,
        reference: str,
        context,
        feedback,
        name: str = None,
        position: int = 0,
        group: str = None,
        color_ramp: str = None,
        ramp_limits: tuple = None,
        ):  # noqa
        """Load and display temporary layer to QGIS project."""
        project = context.project()
        layerTreeRoot = project.layerTreeRoot()
        
        if isinstance(reference, QgsVectorLayer):
            lyr = reference
        else:
            lyr = QgsProcessingUtils.mapLayerFromString(reference, context=context)
        
        if lyr is None:
            feedback.pushWarning('Cannot load layer:  ' + str(reference))
            return
        
        # set layer name
        if name is not None:
            lyr.setName(name)
        
        # apply `color_ramp` to QgsRasterLayer
        if isinstance(lyr, QgsRasterLayer) and color_ramp is not None:
            vmin, vmax = ramp_limits if ramp_limits is not None else (None, None)
            set_SingleBandPseudoColor(lyr, color_ramp, vmin, vmax)
        
        project.addMapLayer(lyr, False)  # True
        
        # Add layer to Qgis layer tree
        if group is not None:
            existing_group = layerTreeRoot.findGroup(group)
            # if not existing_group:
            #     _group = layerTreeRoot.insertGroup(0, group)
            #     _group.insertChildNode(position, QgsLayerTreeLayer(lyr))
            # else:
            #     existing_group.insertChildNode(position, QgsLayerTreeLayer(lyr))
            if not existing_group:
                existing_group = layerTreeRoot.insertGroup(0, group)
            node = QgsLayerTreeLayer(lyr)
            node.setExpanded(False)
            existing_group.insertChildNode(position, node)
        else:
            layerTreeRoot.insertChildNode(position, QgsLayerTreeLayer(lyr))
        
        return lyr
    
    def postProcessAlgorithm(self, context, feedback):
        """
        Apply post-processing steps.
        
        """
        LAYER_NAMES = {
            self.OUTPUT_DEM_DEPRESSIONS: 'Depression mask',
            self.OUTPUT_BPI: 'BPI',
            self.OUTPUT_BPI_MASK: 'BPI mask (processed)',
            self.OUTPUT: 'Pockmarks (delineated)',
            self.OUTPUT_LOG: 'Pockmark delineation parameters',
        }
        
        color_ramp = 'RdBu'
        ramp_limits_BPI = (-5, 5)
        
        # Set ouput group name
        bpi_inner, bpi_outer = self.GROUP_SUFFIX
        group_name = 'Pockmark delineation (BPI {i}-{o}) - {name}'.format(
            i=int(bpi_inner), o=int(bpi_outer), name=self.SOURCE_NAME
        )
        
        # Load intermediate layers
        if self.DEBUG:
            group_name_debug = '[DEBUG] ' + group_name
            
            # Create group
            project = context.project()
            root = project.layerTreeRoot()
            _ = root.insertGroup(0, group_name_debug)
            
            kwargs = dict(context=context, feedback=feedback, group=group_name_debug)
            
            if self.BPI_SMOOTHING:
                self.load_temporary_layer(self.OUTPUTS['BPI'], name='BPI (calculated)', color_ramp=color_ramp, ramp_limits=ramp_limits_BPI, **kwargs)
            else:
                self.load_temporary_layer(self.OUTPUTS['BPI_smoothed'], name='BPI (smoothed)', color_ramp=color_ramp, ramp_limits=ramp_limits_BPI, **kwargs)
            self.load_temporary_layer(self.OUTPUTS['BPI_mask_threshold'], name='BPI mask (threshold)', **kwargs)
            # NTBP! self.load_temporary_layer(self.OUTPUTS['BPI_mask_morph_fill_holes'], name='BPI mask (filled holes)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['BPI_mask_buffer'], name='BPI mask (buffered)', **kwargs)
            
            self.load_temporary_layer(self.OUTPUTS['DEM_clipped_bpi'], name='DEM (clip)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['DEM_clipped_bpi_buffer'], name='DEM (clip, buffer)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['DEM_fenced'], name='DEM (fenced)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['DEM_filled'], name='DEM (filled depressions)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['DEM_depressions_mask_buffered'], name='DEM depression mask', **kwargs)
            
            self.load_temporary_layer(self.OUTPUTS['polygons_BPI'], name='BPI depressions (polygonized)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['polygons'], name='Depressions (polygonized)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['polygons_filtered'], name='Depressions (filtered)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['polygons_buffered'], name='Depressions (filtered, buffered)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['raster_polygons_buffered'], name='Rasterized depressions (filtered, buffered)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['raster_polygons_buffered_clip'], name='Rasterized depressions (filtered, buffered, clip)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['polygons_buffered_clip'], name='Depressions (filtered, buffered, clip)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['polygons_buffered_clip_valid'], name='Depressions (filtered, buffered, clip, valid)', **kwargs)
            
            self.load_temporary_layer(self.OUTPUTS['polygons_smoothed_id'], name='Depressions (smoothed)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['polygons_vertices'], name='Depressions vertices', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['polygons_vertices_append'], name='Depressions vertices (extended)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['voronoi_polygons'], name='Voronoi polygons', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['voronoi_polygons_dissolved'], name='Voronoi polygons (dissolved by FID)', **kwargs)
            
            self.load_temporary_layer(self.OUTPUTS['polygons_buffered_clipped'], name='Depressions (non-overlap buffer)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['polygons_buffered_clipped_filtered'], name='Depressions (non-overlap buffer, no duplicates)', **kwargs)
            
            # self.load_temporary_layer(self.OUTPUTS['polygons_pockmarks'], name='Pockmarks (BPI intersets DEM depressions)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['_polygons_pockmarks_buffer_inv'], name='_polygons_pockmarks_buffer_inv', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['_polygons_pockmarks_simple'], name='_polygons_pockmarks_simple', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['_polygons_pockmarks_buffer'], name='_polygons_pockmarks_buffer', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['_polygons_pockmarks_simple2'], name='_polygons_pockmarks_simple2', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['_polygons_pockmarks_geom'], name='_polygons_pockmarks_geom', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['polygons_pockmarks_output'], name='Pockmarks (simplified, smoothed)', **kwargs)
            
            self.load_temporary_layer(self.OUTPUTS['MBG_polygons_pockmarks_stats'], name='MBG Pockmarks (stats)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['polygons_pockmarks_stats'], name='Pockmarks (stats)', **kwargs)
            
        # Post-process algorithm outputs
        kwargs_postprocessor = {
            self.OUTPUT: dict(group_name=group_name, opacity=0.6),
            self.OUTPUT_BPI: dict(group_name=group_name, ramp=color_ramp, ramp_limits=ramp_limits_BPI),
        }
        
        for result_name, result_id in self.RESULTS.items():
            # if self.DEBUG:
            #     print(result_name, result_id)
            if context.willLoadLayerOnCompletion(result_id):
                # if self.DEBUG:
                #     print('Will load:', result_name)
                details = context.layerToLoadOnCompletionDetails(result_id)
                
                # set display name
                details.name = LAYER_NAMES.get(result_name, result_name)
                
                # set postProcessor (group)
                self.POST_PROCESSORS[result_id] = init_CustomPostProcessor(
                    **kwargs_postprocessor.get(result_name, dict(group_name=group_name))
                )
                details.setPostProcessor(self.POST_PROCESSORS[result_id])
        
        return {}


def init_CustomPostProcessor(
    group_name: str,
    opacity: float = None,
    ramp: str = None,
    ramp_limits: tuple = None
    ):  # noqa
    """
    Wrap PostProcessor class.
    
    Source
    ------
        https://gis.stackexchange.com/a/466554
    
    """
    class CustomPostProcessor(QgsProcessingLayerPostProcessorInterface):
        
        instance = None
        GROUP_NAME = group_name

        def postProcessLayer(self, layer, context, feedback):
            if not isinstance(layer, (QgsRasterLayer, QgsVectorLayer)):
                return
            
            if isinstance(layer, QgsRasterLayer):
                if ramp is not None:
                    vmin, vmax = ramp_limits
                    set_SingleBandPseudoColor(layer, ramp, vmin, vmax)
                    
            if opacity is not None:
                layer.setOpacity(0.6)
            
            project = context.project()
            layerTreeRoot = project.layerTreeRoot()
            
            if not layerTreeRoot.findGroup(self.GROUP_NAME):
                layerTreeRoot.insertGroup(0, self.GROUP_NAME)
            
            group = layerTreeRoot.findGroup(self.GROUP_NAME)
            lyr_node = layerTreeRoot.findLayer(layer.id())
            if lyr_node:
                node_clone = lyr_node.clone()
                node_clone.setExpanded(False)
                group.insertChildNode(0, node_clone)
                lyr_node.parent().removeChildNode(lyr_node)
        
        # Hack to work around sip bug!
        @staticmethod
        def create() -> 'CustomPostProcessor':
            """
            Return a new instance of the post processor.
            Keeping a reference to the sip wrapper so that sip doesn't get confused with the Python subclass and call
            the base wrapper implementation instead...
            """
            CustomPostProcessor.instance = CustomPostProcessor()
            return CustomPostProcessor.instance
            
    return CustomPostProcessor.create()
