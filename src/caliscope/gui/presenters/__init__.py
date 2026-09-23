"""Presenters for MVP architecture.

Presenters coordinate workflow state and adapt domain logic for Views.
They hold transient "scratchpad" state; results aren't persisted until
emitted to the Controller.
"""
