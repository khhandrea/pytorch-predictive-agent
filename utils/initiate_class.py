def get_class_from_module(module_name: str, class_name: str) -> object:
    try:
        module = __import__(module_name, fromlist=(class_name,))
        cls = getattr(module, class_name)
        return cls
    except AttributeError:
        raise ValueError(f"Class {class_name} does not exist.")