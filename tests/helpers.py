from collections import defaultdict

import jaxrts


def get_all_models():
    """
    Get a list of all models defined within jaxrts
    """
    all_models = defaultdict(list)

    for module in [jaxrts.models]:
        for obj_name in dir(module):
            if "__class__" in dir(obj_name):
                attributes = getattr(module, obj_name)
                if "allowed_keys" in dir(attributes):
                    keys = attributes.allowed_keys
                    if (
                        ("Model" not in obj_name)
                        & ("model" not in obj_name)
                        & (not obj_name.startswith("_"))
                    ):
                        for k in keys:
                            all_models[k].append(getattr(module, obj_name))

    return all_models
