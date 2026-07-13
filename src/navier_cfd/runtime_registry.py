from __future__ import annotations


NATIVE_REFERENCE_MODEL_IDS = frozenset(
    {
        "pinn",
        "nsfnets",
        "pinnsformer",
        "deeponet",
        "mionet",
        "fourier_deeponet",
        "nested_fourier_deeponet",
        "fourier_mionet",
        "fno",
        "pino",
        "geo_fno",
        "gino",
        "u_fno",
        "f_fno",
        "u_no",
        "lsm",
        "gnot",
        "galerkin_transformer",
        "mwt",
        "factformer",
        "ono",
        "transolver",
        "upt",
        "meshgraphnets",
        "domino",
        "pibert",
        "fourierflow",
        "pde_refiner",
        "dpot",
        "poseidon",
        "prose_fd",
        "bcat",
        "pdeformer1",
        "pi_mfm",
        "laplace_no",
        "state_space_no",
        "p3d",
        "aerotransformer",
        "tadpole",
        "solver_in_loop",
        "inc",
        "neurosem",
        "np_newton",
        "geometry_preconditioner",
        "revit",
        "deepmmnet",
        "conformal_deeponet",
        "tante",
        "riemannonet",
        "energy_transformer",
        "fun_diff",
        "flow_matching_pde",
    }
)

SPECIALIZED_EXTERNAL_MODEL_IDS = frozenset({"pict", "diffsph", "neuraldem"})


__all__ = ["NATIVE_REFERENCE_MODEL_IDS", "SPECIALIZED_EXTERNAL_MODEL_IDS"]
