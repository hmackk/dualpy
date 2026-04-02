from collections.abc import Callable

UFUNC_REGISTRY: dict[str, Callable] = {}
FUNC_REGISTRY: dict[str, Callable] = {}


def register_ufunc(name: str) -> Callable[[Callable], Callable]:
    def decorator(ufunc: Callable) -> Callable:
        UFUNC_REGISTRY[name] = ufunc
        return ufunc

    return decorator


def register_func(name: str) -> Callable[[Callable], Callable]:
    def decorator(func: Callable) -> Callable:
        FUNC_REGISTRY[name] = func
        return func

    return decorator
