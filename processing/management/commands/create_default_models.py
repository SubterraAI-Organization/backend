from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.core.files import File
from processing.models import Model
from segmentation.models.model_types import ModelType
from pathlib import Path
import os

class Command(BaseCommand):
    help = 'Creates default AI models in the database'

    def handle(self, *args, **kwargs):
        admin_user, created = User.objects.get_or_create(
            username='admin',
            defaults={
                'is_staff': True,
                'is_superuser': True,
                'email': 'admin@example.com'
            }
        )
        
        if created:
            admin_user.set_password('admin')
            admin_user.save()
            self.stdout.write(self.style.SUCCESS('Created default admin user'))

        if not Model.objects.filter(name='UNET').exists():
            unet_weights_path = Path('segmentation/models/saved_models/unet_saved.pth')

            if unet_weights_path.exists():
                with open(unet_weights_path, 'rb') as f:
                    model = Model(
                        name='UNET',
                        description='U-Net architecture for semantic segmentation of root images',
                        model_type=ModelType.UNET.value,
                        public=True,
                        owner=admin_user
                    )
                    model.model_weights.save('unet_saved.pth', File(f))
                    model.save()
                self.stdout.write(self.style.SUCCESS('Successfully created UNET model with weights'))
            else:
                self.stdout.write(self.style.WARNING(f'UNET weights file not found at {unet_weights_path}'))
                Model.objects.create(
                    name='UNET',
                    description='U-Net architecture for semantic segmentation of root images',
                    model_type=ModelType.UNET.value,
                    public=True,
                    owner=admin_user
                )
                self.stdout.write(self.style.WARNING('Created UNET model without weights'))
        else:
            self.stdout.write(self.style.WARNING('UNET model already exists'))

        if not Model.objects.filter(name='YOLO').exists():
            yolo_weights_path = Path('segmentation/models/saved_models/yolo_saved.pt')

            if yolo_weights_path.exists():
                with open(yolo_weights_path, 'rb') as f:
                    model = Model(
                        name='YOLO',
                        description='YOLO object detection model adapted for root segmentation',
                        model_type=ModelType.YOLO.value,
                        public=True,
                        owner=admin_user
                    )
                    model.model_weights.save('yolo_saved.pt', File(f))
                    model.save()
                self.stdout.write(self.style.SUCCESS('Successfully created YOLO model with weights'))
            else:
                self.stdout.write(self.style.WARNING(f'YOLO weights file not found at {yolo_weights_path}'))
                Model.objects.create(
                    name='YOLO',
                    description='YOLO object detection model adapted for root segmentation',
                    model_type=ModelType.YOLO.value,
                    public=True,
                    owner=admin_user
                )
                self.stdout.write(self.style.WARNING('Created YOLO model without weights'))
        else:
            self.stdout.write(self.style.WARNING('YOLO model already exists'))