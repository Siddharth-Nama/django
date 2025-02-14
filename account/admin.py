# admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser
from django.utils.translation import gettext_lazy as _

class CustomUserAdmin(UserAdmin):
    # Define fields to display in the admin panel
    list_display = ('email', 'first_name', 'last_name', 'user_type', 'is_staff', 'is_active', 'created_at', 'updated_at')
    list_filter = ('user_type', 'is_staff', 'is_superuser', 'is_active')
    search_fields = ('email', 'first_name', 'last_name')
    ordering = ('email',)

    # Specify the form layout for creating/editing users
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'user_type')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
    )

    # Fields shown when creating a new user
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'first_name', 'last_name', 'password1', 'password2', 'is_staff', 'is_active', 'user_type'),
        }),
    )

# Register the CustomUser model with CustomUserAdmin configuration
admin.site.register(CustomUser, CustomUserAdmin)

# Customize admin site header and title
admin.site.site_header = _("IIITBH SGC Election Portal")
admin.site.site_title = _("IIITBH SGC Election Portal Admin")
admin.site.index_title = _("Welcome to the IIITBH SGC Election Portal Admin Dashboard")
