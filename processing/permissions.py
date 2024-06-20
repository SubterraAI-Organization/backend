from rest_framework import permissions
from django.http import HttpRequest
from django.views import View

from processing.models import Dataset, Picture, Mask


class IsOwnerOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request: HttpRequest, view: View, obj: type[Dataset | Picture | Mask]):
        if obj.owner == request.user:
            return True
        elif obj.public and request.method in permissions.SAFE_METHODS:
            return True
        else:
            return False
