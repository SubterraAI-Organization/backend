from rest_framework import serializers
from rest_framework.serializers import ImageField, FloatField, IntegerField, PrimaryKeyRelatedField, FileField, CharField
from processing.models import Dataset, Picture, Mask, Model


class DatasetSerializer(serializers.ModelSerializer):
    pictures = PrimaryKeyRelatedField(many=True, read_only=True)

    class Meta:
        model = Dataset
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'owner', 'pictures']


class PictureSerializer(serializers.ModelSerializer):
    image = ImageField()

    class Meta:
        model = Picture
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'dataset']


class MaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = Mask
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'image', 'mask', 'picture', 'root_count',
                            'average_root_diameter', 'total_root_length', 'total_root_area', 'total_root_volume',
                            'original_image', 'overlay_image', 'mask_image']
        write_only_fields = ['threshold']
        extra_kwargs = {
            'threshold': {'required': False, 'default': 0},
        }


class LabelMeSerializer(serializers.ModelSerializer):
    json = FileField()

    class Meta:
        model = Mask
        fields = ['json']
        read_only_fields = ['created', 'updated', 'image', 'mask', 'picture']
        write_only_fields = ['json']
        extra_kwargs = {
            'json': {'required': True},
        }

    def create(self, validated_data) -> Mask:
        return Mask.objects.create(image=validated_data['image'], picture=validated_data['picture'], threshold=0)


class ModelSerializer(serializers.ModelSerializer):
    model_weights = FileField()

    class Meta:
        model = Model
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'owner']
        write_only_fields = ['model_weights', 'model_type']
        extra_kwargs = {'model_weights': {'required': True}}


class AnalysisSerializer(serializers.Serializer):
    image = ImageField(required=False)
    scaling_factor = FloatField(required=False)


class SegmentationSerializer(serializers.Serializer):
    image = ImageField(required=False)
    threshold = IntegerField(required=False, default=25)


class RefinementPredictionSerializer(serializers.Serializer):
    model_type = CharField(required=False, default='unet')
    confidence_threshold = FloatField(required=False)
    area_threshold = IntegerField(required=False, default=0)
    use_refinement = serializers.BooleanField(required=False, default=False)
    refinement_method = CharField(required=False, default='additive')
    
    def validate_model_type(self, value):
        valid_types = ['unet', 'yolo']
        if value.lower() not in valid_types:
            raise serializers.ValidationError(f"model_type must be one of: {valid_types}")
        return value.lower()
    
    def validate_refinement_method(self, value):
        valid_methods = ['additive', 'max', 'weighted']
        if value.lower() not in valid_methods:
            raise serializers.ValidationError(f"refinement_method must be one of: {valid_methods}")
        return value.lower()


class MultiImageMaskSerializer(serializers.ModelSerializer):
    """Enhanced mask serializer that includes all image outputs"""
    
    class Meta:
        model = Mask
        fields = [
            'id', 'picture', 'threshold', 'created', 'updated',
            'use_refinement', 'refinement_method',
            'original_image', 'overlay_image', 'mask_image', 'image',
            'root_count', 'average_root_diameter', 'total_root_length', 
            'total_root_area', 'total_root_volume', 'average_root_length'
        ]
        read_only_fields = [
            'id', 'created', 'updated', 'image', 'picture',
            'original_image', 'overlay_image', 'mask_image',
            'root_count', 'average_root_diameter', 'total_root_length', 
            'total_root_area', 'total_root_volume', 'average_root_length'
        ]


class BulkPredictionSerializer(serializers.Serializer):
    """Serializer for bulk prediction requests with refinement support"""
    ids = CharField(help_text="Comma-separated list of image IDs")
    model_type = CharField(required=False, default='unet')
    confidence_threshold = FloatField(required=False)
    area_threshold = IntegerField(required=False, default=0)
    use_refinement = serializers.BooleanField(required=False, default=False)
    refinement_method = CharField(required=False, default='additive')
    
    def validate_model_type(self, value):
        valid_types = ['unet', 'yolo']
        if value.lower() not in valid_types:
            raise serializers.ValidationError(f"model_type must be one of: {valid_types}")
        return value.lower()
    
    def validate_refinement_method(self, value):
        valid_methods = ['additive', 'max', 'weighted']
        if value.lower() not in valid_methods:
            raise serializers.ValidationError(f"refinement_method must be one of: {valid_methods}")
        return value.lower()
        
    def validate_ids(self, value):
        if not value:
            raise serializers.ValidationError("IDs parameter is required")
        try:
            ids = [int(id.strip()) for id in value.split(',') if id.strip()]
            if not ids:
                raise serializers.ValidationError("At least one valid ID is required")
            return ids
        except ValueError:
            raise serializers.ValidationError("All IDs must be valid integers")
