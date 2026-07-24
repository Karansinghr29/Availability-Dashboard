"""Availability AI Recommendation System — source package.

Only :mod:`data_loader` is aware of the data source. Every other module works
purely on pandas DataFrames, so the future CSV -> PostgreSQL/Supabase migration
touches ``data_loader.py`` alone.
"""
