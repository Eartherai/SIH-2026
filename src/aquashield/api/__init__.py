"""FastAPI service. Import `app` lazily so the core package has no web dependency."""
__all__ = ["app"]


def __getattr__(name):
    if name == "app":
        from .app import app
        return app
    raise AttributeError(name)
