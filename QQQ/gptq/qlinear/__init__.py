def dynamically_import_QuantLinear(
    bits: int,
    disable_marlin: bool = True,
):
    if bits == 4 and not disable_marlin:
        from .qlinear_marlin import QuantLinear
    else:
        from .qlinear_cuda import QuantLinear
    return QuantLinear