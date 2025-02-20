"""
Characterize pockmarks in input DEM using delineated polygons.

"""
import os
from math import sqrt, ceil
from pprint import pprint
from textwrap import dedent

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingUtils,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterBand,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterDefinition,
    QgsProcessingMultiStepFeedback,
    QgsLayerTreeLayer,
    QgsField,
    QgsExpression,
    QgsExpressionContext,
    QgsExpressionContextUtils,
    QgsProcessingLayerPostProcessorInterface,
    QgsRasterLayer,
    QgsVectorLayer,
    edit,
    QgsFeatureRequest,
)
from qgis import processing
from osgeo import gdal, osr, ogr  # noqa


def load_temporary_layer(
    reference: str,
    context,
    feedback,
    name: str = None,
    position: int = 0,
    group: str = None,
):
    """Load and display temporary layer to QGIS project."""
    lyr = QgsProcessingUtils.mapLayerFromString(reference, context=context)
    if lyr is not None:
        if name is not None:
            lyr.setName(name)
        context.project().addMapLayer(lyr, False)  # True
        context.project().layerTreeRoot().insertChildNode(position, QgsLayerTreeLayer(lyr))
    else:
        feedback.pushWarning('Cannot load layer:  ' + str(reference))
        return
    
    return lyr


def set_nodata_geotiff(path: str, band: int = 1, nodata: float = 0.0, calc_stats: bool = True) -> None:
    """Set NoData value for GeoTIFF file using GDAL."""
    ds = gdal.Open(path, gdal.GA_Update)
    band = ds.GetRasterBand(band)
    band.SetNoDataValue(nodata)
    if calc_stats:
        band.ComputeStatistics(approx_ok=False)
    ds = band = None
    return

class PockmarkCharacterization(QgsProcessingAlgorithm):
    """
    Characterize pockmark polygons (depressions) using input DEM.
    
    """
    
    # INPUTS
    INPUT_DEM = 'INPUT_DEM'
    BAND = 'BAND'
    INPUT_POCKMARKS = 'INPUT_POCKMARKS'
    
    COMPUTE_CENTROIDS = 'COMPUTE_CENTROIDS'
    COMPUTE_LOCAL_MINIMA = 'COMPUTE_LOCAL_MINIMA'
    
    # PARAMETERS
    DEBUG_MODE = 'DEBUG_MODE'
    MINIMUM_FILTER_SIZE = 'MINIMUM_FILTER_SIZE'
    LOCAL_MINIMA_BUFFER_SIZE = 'LOCAL_MINIMA_BUFFER_SIZE'

    # OUTPUTS
    OUTPUT = 'OUTPUT'
    OUTPUT_CENTROIDS = 'OUTPUT_CENTROIDS'
    OUTPUT_LOCAL_MINIMA = 'OUTPUT_LOCAL_MINIMA'
    
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
        return PockmarkCharacterization()

    def name(self):  # noqa
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'pockmarkcharacterization'

    def displayName(self):  # noqa
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr('(2) Pockmark characterization')

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
        return self.tr('Characterize pockmark polygons using several computed statistical attributes')
    
    def printFeedback(self, func, msg: str, prefix=f'\n{80 * "="}\n', suffix=f'\n{80 * "="}'):
        """Wrap feedback.push* functions."""
        func(self.tr(prefix + msg + suffix))
    
    def generateTempFilename(self, filename, context):
        """
        Generate temporary filename using processing algorithm context.
        
        """
        # tmp_folder = context.temporaryFolder()
        # if tmp_folder != '':
        #     path = os.path.join(tmp_folder, filename)
        # else:
        #     path = QgsProcessingUtils.generateTempFilename(filename, context)
        
        return QgsProcessingUtils.generateTempFilename(filename, context)

    def initAlgorithm(self, config=None):
        """
        Define the algorithm inputs and outputs of the algorithm as well as additional parameters.
        
        """
        # INPUT
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                name=self.INPUT_DEM,
                description=self.tr('Input DEM'),
                defaultValue=None,
                optional=False
            )
        )
        self.addParameter(
            QgsProcessingParameterBand(
                name=self.BAND,
                description=self.tr('Band number'),
                defaultValue=1,
                parentLayerParameterName=self.INPUT_DEM,
                optional=False,
                allowMultiple=False
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                name=self.INPUT_POCKMARKS,
                description=self.tr('Pockmark polygons (delineateed)'),
                types=[QgsProcessing.TypeVectorPolygon],
                defaultValue=None,
                optional=False
            )
        )
        
        # PARAMETERS
        self.addParameter(
            QgsProcessingParameterBoolean(
                name=self.COMPUTE_CENTROIDS,
                description=self.tr('Extract pockmark centroids?'),
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                name=self.COMPUTE_LOCAL_MINIMA,
                description=self.tr('Extract points of local minima within pockmarks?'),
                defaultValue=True,
            )
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
        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.MINIMUM_FILTER_SIZE,
                description=self.tr('<b>[Local Minima]</b> Filter dimensions (grid cells) of <i>Minimum Filter</i> (should be odd integer)'),
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=0,
            )
        )
        self.parameterDefinition(self.MINIMUM_FILTER_SIZE).setFlags(
            QgsProcessingParameterDefinition.FlagAdvanced
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                name=self.LOCAL_MINIMA_BUFFER_SIZE,
                description=self.tr('<b>[Local Minima]</b> Buffer size (m) to dissolve neighbouring minima within intra-pockmark depression (-1: based on DEM cell size)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=-1,
                minValue=-1,
            )
        )
        self.parameterDefinition(self.LOCAL_MINIMA_BUFFER_SIZE).setFlags(
            QgsProcessingParameterDefinition.FlagAdvanced
        )
        
        # OUTPUT
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                name=self.OUTPUT,
                description='Pockmarks (described)',
                type=QgsProcessing.TypeVectorPolygon,
                createByDefault=True,
                supportsAppend=True,
                defaultValue=None
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                name=self.OUTPUT_CENTROIDS,
                description='Pockmark centroids',
                type=QgsProcessing.TypeVectorPoint,
                createByDefault=False,
                supportsAppend=True,
                defaultValue=None
            )
        )
        self.parameterDefinition(self.OUTPUT_CENTROIDS).setFlags(
            QgsProcessingParameterDefinition.FlagOptional
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                name=self.OUTPUT_LOCAL_MINIMA,
                description='Pockmark points of local minima',
                type=QgsProcessing.TypeVectorPoint,
                createByDefault=True,
                supportsAppend=True,
                defaultValue=None
            )
        )
        self.parameterDefinition(self.OUTPUT_LOCAL_MINIMA).setFlags(
            QgsProcessingParameterDefinition.FlagOptional
        )

    def processAlgorithm(self, parameters, context, model_feedback):
        """
        Characterize pockmarks.
        """
        feedback = QgsProcessingMultiStepFeedback(12, model_feedback)
        
        # init mandatory output (will be lowermost node in group)
        self.RESULTS[self.OUTPUT] = None
        
        # set default prefix
        PREFIX_DESC = 'DESC_'
        
        # ==========================================================================================

        DEM = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)
        self.DEM_NAME = DEM.name()
        BAND = self.parameterAsInt(parameters, self.BAND, context)
        # POCKMARKS = self.parameterAsVectorLayer(parameters, self.INPUT_POCKMARKS, context)
        POCKMARKS_SRC = self.parameterAsSource(parameters, self.INPUT_POCKMARKS, context)
        POCKMARKS = POCKMARKS_SRC.materialize(QgsFeatureRequest())
        
        self.GET_CENTROIDS = self.parameterAsBoolean(parameters, self.COMPUTE_CENTROIDS, context)
        self.GET_LOCAL_MINIMA = self.parameterAsBoolean(parameters, self.COMPUTE_LOCAL_MINIMA, context)
        self.DEBUG = self.parameterAsBoolean(parameters, self.DEBUG_MODE, context)
        SIZE_MIN_FILTER = self.parameterAsInt(parameters, self.MINIMUM_FILTER_SIZE, context)
        BUFFER_LOCAL_MINIMA = self.parameterAsDouble(parameters, self.LOCAL_MINIMA_BUFFER_SIZE, context)
        if BUFFER_LOCAL_MINIMA == -1:
            BUFFER_LOCAL_MINIMA = ceil(sqrt(DEM.rasterUnitsPerPixelX()**2 + DEM.rasterUnitsPerPixelY()**2))
        
        if self.DEBUG:
            print('\n' + '=' * 80)
            pprint(parameters, sort_dicts=False)
            print('POCKMARKS_SRC:', POCKMARKS_SRC)
            print('POCKMARKS:', POCKMARKS)
        
        # ==========================================================================================
        
        # Create temporary ID field
        FIELD_UNIQUE_ID = '_id'
        field_id = [f.name() for f in POCKMARKS.fields() if f.name() == FIELD_UNIQUE_ID]
        HAS_FID = (len(field_id) > 0)
        if not HAS_FID:
            self.printFeedback(feedback.pushDebugInfo, '[POCKMARKS] Create unique `fid` field')
            __POCKMARKS = processing.run('native:fieldcalculator', {
                'INPUT': POCKMARKS,
                'FIELD_NAME': FIELD_UNIQUE_ID,
                'FORMULA': '$id',
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['OUTPUT']
        
        self.printFeedback(feedback.pushInfo, '[POCKMARKS] Rasterize pockmark polygons')
        self.OUTPUTS['POCKMARKS_RASTERIZED'] = processing.run('gdal:rasterize', {
            'INPUT': POCKMARKS if HAS_FID else __POCKMARKS,
            'FIELD': FIELD_UNIQUE_ID,
            'UNITS': 1,  # Georeferenced units
            'WIDTH': DEM.rasterUnitsPerPixelX(),
            'HEIGHT': DEM.rasterUnitsPerPixelY(),
            'EXTENT': DEM,
            'NODATA': -1,
            'DATA_TYPE': 1,  # Int16
            'OUTPUT': QgsProcessingUtils.generateTempFilename('POCKMARKS_RASTERIZED.tif')
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        # self.printFeedback(feedback.pushInfo, '[DEM] Fill depressions')
        # self.OUTPUTS['DEM_FILLED_DEPRESSIONS'] = processing.run('wbt:FillDepressions', {
        #     'dem': DEM,
        #     'fix_flats': False,
        #     'output': QgsProcessingUtils.generateTempFilename('DEM_FILLED_DEPRESSIONS.tif')
        #     },
        #     is_child_algorithm=True,
        #     context=context,
        #     feedback=feedback,
        # )['output']
        
        self.printFeedback(feedback.pushInfo, '[DEM] Fill depressions')
        self.OUTPUTS['DEM_FILLED_DEPRESSIONS'] = processing.run('sagang:fillsinkswangliu', {
            'ELEV': DEM,
            'MINSLOPE': 0.01,
            'FDIR': 'none',
            'WSHED': 'none',
            'FILLED': QgsProcessingUtils.generateTempFilename('DEM_FILLED_DEPRESSIONS.tif')
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['FILLED']
        
        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        self.printFeedback(feedback.pushInfo, '[DEM] Clip input DEM to pockmark polygons (relative depths)')
        self.OUTPUTS['DEM_POCKMARKS_DEPTHS'] = processing.run('gdal:rastercalculator', {
            'INPUT_A': DEM,
            'BAND_A': 1,
            'INPUT_B': self.OUTPUTS['DEM_FILLED_DEPRESSIONS'],
            'BAND_B': 1,
            'INPUT_C': self.OUTPUTS['POCKMARKS_RASTERIZED'],
            'BAND_C': 1,
            'FORMULA': '(A - B) * (C > 0)',
            'NO_DATA': 0.0,
            'RTYPE': 5,  # 5: Float32
            'OUTPUT': QgsProcessingUtils.generateTempFilename('DEM_POCKMARKS_DEPTHS.tif'),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}

        self.printFeedback(feedback.pushInfo, '[POCKMARKS] Compute descriptive pockmark attributes')
        FORMULA = 'geom_to_wkt(centroid($geometry))'
        FIELD_CENTROID = 'GEOM_centroid'
        self.OUTPUTS['POCKMARKS_CENTROID'] = processing.run('native:fieldcalculator', {
            'INPUT': POCKMARKS,
            'FIELD_NAME': FIELD_CENTROID,
            'FIELD_TYPE': 2,  # Text
            'FIELD_LENGTH': 100,
            'FORMULA': FORMULA,
            'OUTPUT': QgsProcessingUtils.generateTempFilename('POCKMARKS_CENTROID.gpkg'),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        feedback.setCurrentStep(4)
        if feedback.isCanceled():
            return {}
        
        # --------------------------------------------------------------------------------------------------------------
        # HOUSEKEEPING - Delete fields to re-compute
        # --------------------------------------------------------------------------------------------------------------
        
        lyr_pockmarks = QgsVectorLayer(self.OUTPUTS['POCKMARKS_CENTROID'], 'POCKMARKS_CENTROID')
        lyr_pockmarks_pr = lyr_pockmarks.dataProvider()
        
        fields = lyr_pockmarks.fields()
        field_names = [f.name() for f in fields]
        field_idx_delete = [
            i for i, f in enumerate(field_names)
            if any(c in f for c in ['MBG_', 'SLOPE_', 'DEPTH_', 'GEOM_', PREFIX_DESC]) and f != FIELD_CENTROID  # 'BPI_', 
        ]
        with edit(lyr_pockmarks):
            check = lyr_pockmarks_pr.deleteAttributes(field_idx_delete)
        lyr_pockmarks.updateFields()
        
        lyr_pockmarks.selectAll()
        self.OUTPUTS['POCKMARKS_CLEANED'] = processing.run('native:saveselectedfeatures', {
            'INPUT': lyr_pockmarks,
            'OUTPUT': self.generateTempFilename('POCKMARKS_CLEANED.gpkg', context=context),
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        lyr_pockmarks.removeSelection()
        
        # --------------------------------------------------------------------------------------------------------------
        # POCKMARK STATISTICS - extracted
        # --------------------------------------------------------------------------------------------------------------
        
        # TODO: Remove fields from attribute table before running Zonal Statistics
        STATISTICS = [2, 3, 4, 5, 6, 7]  # Mean, Median, StdDev, Minimum, Maximum, Range
        # (a) on SLOPE
        self.OUTPUTS['DEM_SLOPE'] = processing.run('native:slope', {
            'INPUT': DEM,
            'Z_FACTOR': 1,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        
        self.OUTPUTS['POCKMARKS_STATS'] = processing.run('native:zonalstatistics', {
            'INPUT_RASTER': self.OUTPUTS['DEM_SLOPE'],
            'RASTER_BAND': 1,
            'INPUT_VECTOR': self.OUTPUTS['POCKMARKS_CLEANED'],  # self.OUTPUTS['POCKMARKS_CENTROID'],
            'COLUMN_PREFIX': 'SLOPE_',
            'STATISTICS': STATISTICS,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['INPUT_VECTOR']
        
        # (b) on DEM
        self.OUTPUTS['POCKMARKS_STATS'] = processing.run('native:zonalstatistics', {
            'INPUT_RASTER': DEM,
            'RASTER_BAND': BAND,
            'INPUT_VECTOR': self.OUTPUTS['POCKMARKS_CLEANED'],  # self.OUTPUTS['POCKMARKS_STATS'],
            'COLUMN_PREFIX': 'DEPTH_',
            'STATISTICS': STATISTICS,
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['INPUT_VECTOR']
        
        # add DEM statistics (for volume calculations)
        self.OUTPUTS['POCKMARKS_STATS'] = processing.run('native:zonalstatistics', {
            'INPUT_RASTER': self.OUTPUTS['DEM_POCKMARKS_DEPTHS'],
            'RASTER_BAND': BAND,
            'INPUT_VECTOR': self.OUTPUTS['POCKMARKS_CLEANED'],
            'COLUMN_PREFIX': 'DEM_',
            'STATISTICS': [0, 1, 7],  # Count, Sum, Range
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['INPUT_VECTOR']
        
        # --------------------------------------------------------------------------------------------------------------
        # MINIMUM BOUNDING GEOMETRY
        # --------------------------------------------------------------------------------------------------------------
        
        self.printFeedback(feedback.pushInfo, '[MBG] Compute Minimum Bounding Geometry (rectangle by width)')
        self.OUTPUTS['MBG_polygons_pockmarks'] = processing.run('wbt:MinimumBoundingBox', {
            'input': self.OUTPUTS['POCKMARKS_CLEANED'],
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
        
        self.printFeedback(feedback.pushInfo, '[POLYGONS] Spatial join MBG stats with pockmark polygons')
        polygons_stats = processing.run('native:joinattributestable', {
            'INPUT': self.OUTPUTS['POCKMARKS_CLEANED'],
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
        
        # --------------------------------------------------------------------------------------------------------------
        # POCKMARK STATISTICS - calculated
        # --------------------------------------------------------------------------------------------------------------
        
        lyr_pockmarks = QgsVectorLayer(self.OUTPUTS['polygons_pockmarks_stats'], 'POCKMARKS_STATS')
        lyr_pockmarks_pr = lyr_pockmarks.dataProvider()
        
        fields_add = [
            QgsField('MBG_aspect_ratio', QVariant.Double, len=10, prec=5),
            QgsField('GEOM_area', QVariant.Double, len=10, prec=3),
            QgsField('GEOM_perimeter', QVariant.Double, len=10, prec=3),
            QgsField(PREFIX_DESC + 'depth', QVariant.Double, len=10, prec=3),
            QgsField(PREFIX_DESC + 'volume_m3', QVariant.Double, len=15, prec=3),
            QgsField(PREFIX_DESC + 'relief_area_ratio', QVariant.Double, len=10, prec=8),
            QgsField(PREFIX_DESC + 'profile_index', QVariant.Double, len=10, prec=5),
            # QgsField(PREFIX_DESC + 'roundness', QVariant.Double, len=10, prec=5),
            QgsField(PREFIX_DESC + 'compactness', QVariant.Double, len=10, prec=5),
            QgsField(PREFIX_DESC + 'circularity', QVariant.Double, len=10, prec=5),
            QgsField(PREFIX_DESC + 'convexity', QVariant.Double, len=10, prec=5),
            QgsField(PREFIX_DESC + 'solidity', QVariant.Double, len=10, prec=5),
            # QgsField(PREFIX_DESC + 'complexity', QVariant.Double, len=10, prec=5),
            QgsField(PREFIX_DESC + 'eccentricity', QVariant.Double, len=10, prec=5),
            QgsField(PREFIX_DESC + 'roughness_index', QVariant.Double, len=10, prec=5),
            QgsField(PREFIX_DESC + 'dissection_index', QVariant.Double, len=10, prec=5),
        ]
        lyr_pockmarks_pr.addAttributes(fields_add)
        lyr_pockmarks.updateFields()

        context_exp = QgsExpressionContext()
        context_exp.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(lyr_pockmarks))

        with edit(lyr_pockmarks):
            # request = QgsFeatureRequest().setFlags(QgsFeatureRequest.NoGeometry)#.setSubsetOfAttributes(['WDEPTH', 'fid'], layer.fields() )
            for f in lyr_pockmarks.getFeatures():  # request
                context_exp.setFeature(f)
                
                f['MBG_aspect_ratio'] = QgsExpression('round("MBG_width" / "MBG_height", 5)').evaluate(context_exp)
                
                f['GEOM_area'] = QgsExpression('round($area, 3)').evaluate(context_exp)
                f['GEOM_perimeter'] = QgsExpression('round($perimeter, 3)').evaluate(context_exp)
                
                f[PREFIX_DESC + 'depth'] = f['DEM_range']
                f[PREFIX_DESC + 'relief_area_ratio'] = QgsExpression('round("DEPTH_range" / $area, 8)').evaluate(context_exp)
                f[PREFIX_DESC + 'profile_index'] = QgsExpression('round(abs("DEPTH_min" - "DEPTH_mean") / ("DEPTH_range"), 5)').evaluate(context_exp)
                
                # Polsby-Popper: 0 < compactness < 1 (most compact)
                f[PREFIX_DESC + 'compactness'] = QgsExpression('round((4 * pi() * $area) / $perimeter^2, 5)').evaluate(context_exp)
                
                # 0 < circularity < 1 (perfect circle)
                f[PREFIX_DESC + 'circularity'] = QgsExpression('round((4 * pi() * $area) / perimeter(convex_hull($geometry))^2, 5)').evaluate(context_exp)
                
                # (most complex) 0 < convexity < 1 (most compact)
                f[PREFIX_DESC + 'convexity'] = QgsExpression('round(perimeter(convex_hull($geometry)) / $perimeter, 5)').evaluate(context_exp)
                
                # (most complex) 0 < solidity < 1 (most compact)
                f[PREFIX_DESC + 'solidity'] = QgsExpression('round($area / area(convex_hull($geometry)), 5)').evaluate(context_exp)
                
                f[PREFIX_DESC + 'volume_m3'] = QgsExpression(
                    'round("DEM_sum" * -1 * {x} * {y}, 3)'.format(x=DEM.rasterUnitsPerPixelX(), y=DEM.rasterUnitsPerPixelY())
                ).evaluate(context_exp)
                
                lyr_pockmarks.updateFeature(f)
                
        with edit(lyr_pockmarks):
            # request = QgsFeatureRequest().setFlags(QgsFeatureRequest.NoGeometry)#.setSubsetOfAttributes(['WDEPTH', 'fid'], layer.fields() )
            for f in lyr_pockmarks.getFeatures():  # request
                context_exp.setFeature(f)
                
                # # Complexity
                # f[PREFIX_DESC + 'complexity'] = QgsExpression('round("DESC_compactness" * "DESC_circularity" * "DESC_convexity" * "DESC_solidity", 5)').evaluate(context_exp)
                
                # Eccentricity
                f[PREFIX_DESC + 'eccentricity'] = QgsExpression(
                    'round(sqrt(("MBG_height"^2 - "MBG_width"^2) / ("MBG_height"^2)), 5)'
                ).evaluate(context_exp)
                
                # Roughness Index (RI)
                f[PREFIX_DESC + 'roughness_index'] = QgsExpression(dedent(
                    '''\
                    round(
                        with_variable(
                            'r_mean',
                            array_mean(
                                array_foreach(
                                    geometries_to_array(nodes_to_points(densify_by_distance(@geometry, 1), True)),
                                    distance(centroid(@geometry), @element)
                                )
                            ),
                            @r_mean^2 / (area(@geometry) + perimeter(@geometry)^2) * 42.61646
                        ),
                    5)
                    '''
                    )
                ).evaluate(context_exp)
                
                # Dissection Index (DI)
                f[PREFIX_DESC + 'dissection_index'] = QgsExpression('round((perimeter(@geometry)/ (2 * area(@geometry))) * sqrt(area(@geometry)/pi()), 5)').evaluate(context_exp)
                
                lyr_pockmarks.updateFeature(f)
        
        feedback.setCurrentStep(5)
        if feedback.isCanceled():
            return {}
        
        # --------------------------------------------------------------------------------------------------------------
        
        self.printFeedback(feedback.pushInfo, '[HOUSEKEEPING] Remove unused attribute fields from pockmark polygons')
        fields = lyr_pockmarks.fields()
        field_names = [f.name() for f in fields]
        
        # fields_delete = [f for f in field_names if not any(s in f for s in [FIELD_CENTROID, 'DEM_'])]
        fields_delete = {
            i: f
            for i, f in enumerate(field_names)
            if any(s in f for s in [FIELD_UNIQUE_ID, FIELD_CENTROID, 'DEM_'])
        }
        
        with edit(lyr_pockmarks):
            _ = lyr_pockmarks.dataProvider().deleteAttributes(list(fields_delete.keys()))
            lyr_pockmarks.updateFields()
        
        # --------------------------------------------------------------------------------------------------------------
        # POCKMARK CENTROIDS
        # --------------------------------------------------------------------------------------------------------------
        
        if self.GET_CENTROIDS and self.OUTPUT_CENTROIDS in parameters.keys():
            self.OUTPUTS['POINTS_CENTROIDS'] = processing.run('native:centroids', {
                'INPUT': lyr_pockmarks,
                'ALL_PARTS': True,
                'OUTPUT': parameters[self.OUTPUT_CENTROIDS],
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['OUTPUT']
            
            self.RESULTS[self.OUTPUT_CENTROIDS] = self.OUTPUTS['POINTS_CENTROIDS']
        
        elif self.GET_CENTROIDS and self.OUTPUT_CENTROIDS not in parameters.keys():
            self.printFeedback(feedback.pushWarning, '[CENTROIDS] Option enabled but no output file specified! Skipping...')
        elif not self.GET_CENTROIDS and self.OUTPUT_CENTROIDS in parameters.keys():
            self.printFeedback(feedback.pushWarning, '[CENTROIDS] Output file specified but option disabled! Skipping...')
            
        # --------------------------------------------------------------------------------------------------------------
        # EXTRACT VERTICES AT DEEPEST POINTS WITHIN POCKMARKS
        # --------------------------------------------------------------------------------------------------------------
        
        if self.GET_LOCAL_MINIMA and self.OUTPUT_LOCAL_MINIMA in parameters.keys():
            
            # SIZE_MIN_FILTER = 5
            self.printFeedback(feedback.pushInfo, '[DEM] Compute minimum filtered raster for rasterized pockmarks')
            self.OUTPUTS['DEM_MINIMUM_FILTER'] = processing.run('wbt:MinimumFilter', {
                'input': self.OUTPUTS['DEM_POCKMARKS_DEPTHS'],
                'filterx': SIZE_MIN_FILTER,
                'filtery': SIZE_MIN_FILTER,
                'output': QgsProcessingUtils.generateTempFilename('DEM_MINIMUM_FILTER.tif'),
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['output']
            
            feedback.setCurrentStep(6)
            if feedback.isCanceled():
                return {}
                
            self.OUTPUTS['DEM_POCKMARKS_LOCAL_MINIMA'] = processing.run('gdal:rastercalculator', {
                'INPUT_A': self.OUTPUTS['DEM_POCKMARKS_DEPTHS'],
                'BAND_A': 1,
                'INPUT_B': self.OUTPUTS['DEM_MINIMUM_FILTER'],
                'BAND_B': 1,
                'FORMULA': '(A == B) * B',
                'NO_DATA': 0.0,
                'RTYPE': 5,  # 5: Float32
                'OUTPUT': QgsProcessingUtils.generateTempFilename('DEM_POCKMARKS_LOCAL_MINIMA.tif'),
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['OUTPUT']
            
            feedback.setCurrentStep(7)
            if feedback.isCanceled():
                return {}
            
            self.printFeedback(feedback.pushInfo, '[POINTS] Extract local minima within pockmark polygons')
            FIELD_DEM_VALUE = 'VALUE'
            self.OUTPUTS['POINTS_LOCAL_MINIMA'] = processing.run('native:pixelstopoints', {
                'INPUT_RASTER': self.OUTPUTS['DEM_POCKMARKS_LOCAL_MINIMA'],
                'RASTER_BAND': 1,
                'FIELD_NAME': FIELD_DEM_VALUE,
                'OUTPUT': QgsProcessingUtils.generateTempFilename('POINTS_LOCAL_MINIMA.gpkg'),
                # 'OUTPUT': parameters[self.OUTPUT_LOCAL_MINIMA]
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['OUTPUT']
            
            # Buffer points -> merge neighbouring points
            self.OUTPUTS['POINTS_LOCAL_MINIMA_BUFFER'] = processing.run('native:buffer', {
                'INPUT': self.OUTPUTS['POINTS_LOCAL_MINIMA'],
                'DISTANCE': BUFFER_LOCAL_MINIMA,
                'SEGMENTS': 8,
                'DISSOLVE': True,
                'SEPARATE_DISJOINT': True,
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['OUTPUT']
            
            # Extract centroids of amalgamated points
            self.OUTPUTS['POINTS_LOCAL_MINIMA_DISSOLVED'] = processing.run('native:centroids', {
                'INPUT': self.OUTPUTS['POINTS_LOCAL_MINIMA_BUFFER'],
                'ALL_PARTS': True,
                'OUTPUT': QgsProcessingUtils.generateTempFilename('POINTS_LOCAL_MINIMA_DISSOLVED.gpkg'),
                # 'OUTPUT': parameters[self.OUTPUT_LOCAL_MINIMA],
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['OUTPUT']
            
            feedback.setCurrentStep(8)
            if feedback.isCanceled():
                return {}
            
            # Sample input DEM depth (absolute)
            self.printFeedback(feedback.pushInfo, '[POINTS] Sample input raster')
            tmp_local_minima_sampled = processing.run('native:rastersampling', {
                'INPUT': self.OUTPUTS['POINTS_LOCAL_MINIMA_DISSOLVED'],
                'RASTERCOPY': DEM,
                'COLUMN_PREFIX': PREFIX_DESC + 'WATER_DEPTH_',
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['OUTPUT']
            
            # Sample input DEM depth (relative)
            tmp_local_minima_sampled_2 = processing.run('native:rastersampling', {
                'INPUT': tmp_local_minima_sampled,
                'RASTERCOPY': self.OUTPUTS['DEM_POCKMARKS_DEPTHS'],
                'COLUMN_PREFIX': PREFIX_DESC + 'DEPTH_relative_',
                'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT,
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['OUTPUT']
            
            feedback.setCurrentStep(9)
            if feedback.isCanceled():
                return {}
            
            
            # [HOUSEKEEPING} Remove attribute fields
            self.printFeedback(feedback.pushInfo, '[POINTS] Remove unused attribute fields')
            lyr_pts = QgsProcessingUtils.mapLayerFromString(tmp_local_minima_sampled_2, context=context)
            dataprovider = lyr_pts.dataProvider()
            # if self.DEBUG:
            #     print('[lyr_pts] CRS:', lyr_pts.crs())
            lyr_pts.setCrs(POCKMARKS.crs())
            # if self.DEBUG:
            #     print('[lyr_pts] CRS:', lyr_pts.crs())
            
            with edit(lyr_pts):
                # (1) Rename fields
                fields = lyr_pts.fields()
                field_names_rename = [field.name() for field in fields if PREFIX_DESC in field.name()]
                field_names_rename_idx = [dataprovider.fieldNameIndex(fname) for fname in field_names_rename]
                renamer = dict(zip(
                    field_names_rename_idx,  # indices
                    ['_'.join(fname.split('_')[:-1]) for fname in field_names_rename]  # new field names
                ))
                check = dataprovider.renameAttributes(renamer)
                if self.DEBUG:
                    print('[renamer]:', renamer)
                    print('Renamed fields?  ', check)
                
                # (2) Remove fields
                fields = lyr_pts.fields()
                field_names_delete = [field.name() for field in fields if PREFIX_DESC not in field.name()]
                fields_delete_idx = [fields.indexFromName(fname) for fname in field_names_delete]
                check = dataprovider.deleteAttributes(fields_delete_idx)
                if self.DEBUG:
                    print('[field_names_delete]:', field_names_delete)
                    print('Removed fields?  ', check)
            
            lyr_pts.updateFields()
            
            feedback.setCurrentStep(10)
            if feedback.isCanceled():
                return {}
            
            
            self.printFeedback(feedback.pushInfo, '[POINTS] Join polygon attributes with extracted points')
            JOIN_PREFIX = 'POCKMARK_'
            self.OUTPUTS['POINTS_DEEPEST_JOIN'] = processing.run('native:joinattributesbylocation', {
                'INPUT': lyr_pts,
                'PREDICATE': 5,  # are within
                'JOIN': lyr_pockmarks,  # POCKMARKS_centroid,
                # 'JOIN_FIELDS': ['fid', FIELD_CENTROID],
                'METHOD': 1,  # Take attributes of the first matching feature only (one-to-one)
                'DISCARD_NONMATCHING': True,
                'PREFIX': JOIN_PREFIX,
                'OUTPUT': parameters[self.OUTPUT_LOCAL_MINIMA],
                # 'OUTPUT': QgsProcessingUtils.generateTempFilename('POINTS_DEEPEST_JOIN.gpkg'),
                # 'NON_MATCHING': '',  # QgsProcessingUtils.generateTempFilename('POINTS_DEEPEST_NON_MATCHING.gpkg')
                },
                is_child_algorithm=True,
                context=context,
                feedback=feedback,
            )['OUTPUT']
            
            self.RESULTS[self.OUTPUT_LOCAL_MINIMA] = self.OUTPUTS['POINTS_DEEPEST_JOIN']
            
            feedback.setCurrentStep(11)
            if feedback.isCanceled():
                return {}
        
        elif self.GET_LOCAL_MINIMA and self.OUTPUT_LOCAL_MINIMA not in parameters.keys():
            self.printFeedback(feedback.pushWarning, '[LOCAL MINIMA] Option enabled but no output file specified! Skipping...')
        elif not self.GET_LOCAL_MINIMA and self.OUTPUT_LOCAL_MINIMA in parameters.keys():
            self.printFeedback(feedback.pushWarning, '[LOCAL MINIMA] Output file specified but option disabled! Skipping...')        

        # ------------------------------------------------------------------------------------------
        
        # [WORKAROUND] Save pockmark polygons to feature sink
        lyr_pockmarks.selectAll()
        self.OUTPUTS['POCKMARKS_DESC'] = processing.run('native:saveselectedfeatures', {
            'INPUT': lyr_pockmarks,
            'OUTPUT': parameters[self.OUTPUT]
            },
            is_child_algorithm=True,
            context=context,
            feedback=feedback,
        )['OUTPUT']
        lyr_pockmarks.removeSelection()
        
        self.RESULTS[self.OUTPUT] = self.OUTPUTS['POCKMARKS_DESC']
        
        feedback.setCurrentStep(12)
        if feedback.isCanceled():
            return {}
        
        # ==============================================================================================================
        
        self.printFeedback(feedback.pushInfo, '', suffix='')
        
        # set RESULTS
        # self.RESULTS[self.OUTPUT] = self.OUTPUTS['POCKMARKS_DESC']
        # self.RESULTS[self.OUTPUT_CENTROIDS] = self.OUTPUTS['POINTS_CENTROIDS']
        # self.RESULTS[self.OUTPUT_LOCAL_MINIMA] = self.OUTPUTS['POINTS_DEEPEST_SELECT']
        
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
        group: str = None
        ):  # noqa
        """Load and display temporary layer to QGIS project."""
        project = context.project()
        layerTreeRoot = project.layerTreeRoot()
        
        lyr = QgsProcessingUtils.mapLayerFromString(reference, context=context)
        if lyr is not None:
            if name is not None:
                lyr.setName(name)
            project.addMapLayer(lyr, False)  # True
            
        else:
            feedback.pushWarning('Cannot load layer:  ' + str(reference))
            return
        
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
            self.OUTPUT_CENTROIDS: 'Pockmark centroids',
            self.OUTPUT_LOCAL_MINIMA: 'Pockmark points (local minima)',
            self.OUTPUT: 'Pockmarks (characterized)',
        }
        
        # Set ouput group name
        group_name = 'Pockmark characterization - {name}'.format(name=self.DEM_NAME)
        
        # Load intermediate layers
        if self.DEBUG:
            group_name_debug = '[DEBUG] ' + group_name
            
            # Create group
            project = context.project()
            root = project.layerTreeRoot()
            _ = root.insertGroup(0, group_name_debug)
            
            kwargs = dict(context=context, feedback=feedback, group=group_name_debug)
            
            self.load_temporary_layer(self.OUTPUTS['POCKMARKS_RASTERIZED'], name='Pockmarks (rasterized)', **kwargs)
            # self.load_temporary_layer(self.OUTPUTS['DEM_FILLED_DEPRESSIONS'], name='DEM (filled depressions)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['DEM_POCKMARKS_DEPTHS'], name='DEM pockmark depths (relative)', **kwargs)
            self.load_temporary_layer(self.OUTPUTS['POCKMARKS_CLEANED'], name='Pockmarks (cleaned)', **kwargs)
            # self.load_temporary_layer(self.OUTPUTS['POCKMARKS_CENTROID'], name='Pockmarks (added centroid WKT)', **kwargs)
            
            if self.GET_LOCAL_MINIMA:
                self.load_temporary_layer(self.OUTPUTS['DEM_MINIMUM_FILTER'], name='DEM (minimum filtered)', **kwargs)
                self.load_temporary_layer(self.OUTPUTS['DEM_POCKMARKS_LOCAL_MINIMA'], name='DEM (pockmarks local minima)', **kwargs)
                self.load_temporary_layer(self.OUTPUTS['POINTS_LOCAL_MINIMA'], name='Points (local minima)', **kwargs)
                self.load_temporary_layer(self.OUTPUTS['POINTS_LOCAL_MINIMA_BUFFER'], name='Buffer (local minima)', **kwargs)
                self.load_temporary_layer(self.OUTPUTS['POINTS_LOCAL_MINIMA_DISSOLVED'], name='Points (local minima, dissolved)', **kwargs)

        # Post-process algorithm outputs
        # print('\n' + '-' * 80)
        # print('POST-PROCESSOR')
        kwargs_postprocessor = {
            self.OUTPUT: dict(group_name=group_name, opacity=0.6),
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
            
            # if isinstance(layer, QgsRasterLayer):
            #     if ramp is not None:
            #         vmin, vmax = ramp_limits
            #         set_SingleBandPseudoColor(layer, ramp, vmin, vmax)
                    
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
