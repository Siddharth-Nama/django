# admin.py
from django.contrib import admin
from .models import Voter, Position, Candidate, Votes

class VoterAdmin(admin.ModelAdmin):
    list_display = ('admin', 'voted')
    search_fields = ('admin__email', 'admin__first_name', 'admin__last_name', 'roll')
    list_filter = ('voted',)

class PositionAdmin(admin.ModelAdmin):
    list_display = ('name', 'max_vote', 'priority')
    ordering = ('priority',)
    search_fields = ('name',)

class CandidateAdmin(admin.ModelAdmin):
    list_display = ('fullname', 'position')
    search_fields = ('fullname',)
    list_filter = ('position',)

class VotesAdmin(admin.ModelAdmin):
    list_display = ('voter', 'position', 'candidate')
    search_fields = ('voter__admin__email', 'position__name', 'candidate__fullname')
    list_filter = ('position',)

# Register each model with its custom admin configuration
admin.site.register(Voter, VoterAdmin)
admin.site.register(Position, PositionAdmin)
admin.site.register(Candidate, CandidateAdmin)
admin.site.register(Votes, VotesAdmin)
