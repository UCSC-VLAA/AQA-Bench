class Registry():
    def __init__(self, name):
        self._name = name
        self._objs = {}

    def _do_register(self, name, obj):
        assert name not in self._objs, f"An object named {name} is already registered. Current Registry: {self}"

        self._objs[name] = obj

    def register(self, obj=None, name=None):
        if obj is None:
            # return a decorator
            def decorator(func_or_cls):
                nonlocal name
                assert isinstance(name, str) or name is None

                if name is None:
                    name = func_or_cls.__name__

                self._do_register(name, func_or_cls)
                return func_or_cls

            return decorator

        assert isinstance(name, str) or name is None
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def __getitem__(self, name):
        assert isinstance(name, str) or name is None
        assert name in self._objs, f"{name} not found. Current Registry: {self}"

        return self._objs[name]

    def __contains__(self, name):
        return name in self._objs

    def __repr__(self):
        return f"{self._name}({self._objs})"
