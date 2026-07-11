const MODEL_ROWS = [["pinn","Physics-Informed Neural Network",["physics_informed","general_pde_solver","inverse"]],["nsfnets","NSFnets",["physics_informed","specialized"]],["pinnsformer","PINNsFormer",["physics_informed","general_pde_solver"]],["deeponet","DeepONet",["surrogate","general_pde_solver","physics_informed"]],["mionet","MIONet",["surrogate","general_pde_solver"]],["fourier_deeponet","Fourier-DeepONet",["surrogate","specialized"]],["nested_fourier_deeponet","Nested Fourier-DeepONet",["surrogate","specialized","particle_multiphase"]],["fourier_mionet","Fourier-MIONet",["surrogate","specialized","particle_multiphase"]],["fno","Fourier Neural Operator",["surrogate","general_pde_solver"]],["pino","Physics-Informed Neural Operator",["physics_informed","surrogate","general_pde_solver"]],["geo_fno","Geo-FNO",["geometry","surrogate","general_pde_solver"]],["gino","Geometry-Informed Neural Operator",["geometry","surrogate"]],["u_fno","U-FNO",["surrogate","specialized","particle_multiphase"]],["f_fno","Factorized FNO",["surrogate","general_pde_solver"]],["u_no","U-shaped Neural Operator",["surrogate","general_pde_solver"]],["lsm","Latent Spectral Model",["surrogate","general_pde_solver"]],["gnot","General Neural Operator Transformer",["surrogate","general_pde_solver","geometry"]],["galerkin_transformer","Galerkin Transformer",["surrogate","general_pde_solver"]],["mwt","Multiwavelet Transformer",["surrogate","general_pde_solver"]],["factformer","FactFormer",["surrogate","general_pde_solver"]],["ono","Orthogonal Neural Operator",["surrogate","general_pde_solver"]],["transolver","Transolver",["geometry","surrogate","general_pde_solver"]],["upt","Universal Physics Transformer",["foundation","geometry","surrogate","general_pde_solver"]],["meshgraphnets","MeshGraphNets",["geometry","surrogate"]],["domino","DoMINO",["geometry","surrogate","specialized"]],["pibert","PIBERT",["surrogate","physics_informed","specialized"]],["fourierflow","FourierFlow",["surrogate","generative","specialized"]],["pde_refiner","PDE-Refiner",["surrogate","generative","general_pde_solver"]],["dpot","DPOT",["foundation","surrogate","general_pde_solver"]],["poseidon","Poseidon",["foundation","surrogate","general_pde_solver"]],["prose_fd","PROSE-FD",["foundation","surrogate","specialized"]],["bcat","BCAT",["foundation","surrogate","specialized"]],["pdeformer1","PDEformer-1",["foundation","general_pde_solver"]],["pi_mfm","PI-MFM",["foundation","physics_informed","general_pde_solver"]],["laplace_no","Laplace Neural Operator",["surrogate","general_pde_solver"]],["state_space_no","State-Space Neural Operator",["surrogate","general_pde_solver"]],["p3d","P3D",["surrogate","foundation","specialized"]],["aerotransformer","AeroTransformer",["foundation","geometry","specialized"]],["tadpole","Tadpole",["foundation","surrogate"]],["solver_in_loop","Solver-in-the-Loop",["acceleration","physics_informed"]],["inc","Indirect Neural Corrector",["acceleration","physics_informed"]],["pict","PICT",["acceleration","specialized","physics_informed"]],["diffsph","diffSPH",["acceleration","particle_multiphase","specialized"]],["neurosem","NeuroSEM",["acceleration","physics_informed","specialized"]],["np_newton","Neural-Operator Preconditioned Newton",["acceleration","general_pde_solver"]],["geometry_preconditioner","Geometry-Aware Neural Preconditioner",["acceleration","geometry"]],["neuraldem","NeuralDEM",["specialized","particle_multiphase","surrogate"]],["revit","ReViT",["geometry","surrogate"]],["deepmmnet","DeepM&Mnet",["surrogate","specialized","physics_informed"]],["conformal_deeponet","Conformalized-DeepONet",["uncertainty","surrogate"]],["tante","TANTE",["surrogate","general_pde_solver"]],["riemannonet","RiemannONet",["physics_informed","specialized","surrogate"]],["energy_transformer","Energy Transformer Flow Reconstruction",["inverse","surrogate","specialized"]],["fun_diff","FunDiff",["generative","physics_informed","general_pde_solver"]],["flow_matching_pde","Flow Matching for PDEs",["generative","physics_informed","general_pde_solver"]]];
const GEOMETRY = new Set(["aerotransformer", "domino", "geo_fno", "geometry_preconditioner", "gino", "gnot", "meshgraphnets", "revit", "transolver", "upt"]);
const THREED = new Set(["aerotransformer", "deepmmnet", "diffsph", "domino", "energy_transformer", "geometry_preconditioner", "gino", "inc", "neuraldem", "neurosem", "np_newton", "p3d", "pict", "revit", "tadpole", "tante", "upt"]);
const ACCEL = new Set(["diffsph", "geometry_preconditioner", "inc", "neurosem", "np_newton", "pict", "solver_in_loop"]);
const LONG = new Set(["bcat", "dpot", "f_fno", "fno", "fourierflow", "inc", "neuraldem", "p3d", "pde_refiner", "pibert", "pinnsformer", "poseidon", "prose_fd", "solver_in_loop", "tadpole", "tante"]);
const PARTICLE = new Set(["diffsph", "neuraldem"]);
const UQ = new Set(["conformal_deeponet", "fourierflow", "fun_diff"]);
const ARCH = {"pinn":"coordinate network with PDE residuals","deeponet":"branch-trunk operator network","fno":"Fourier neural operator","pibert":"bidirectional transformer with Fourier-wavelet embeddings and physics-biased attention","fourierflow":"frequency-guided generative flow model","transolver":"physics-attention transformer on general geometries","upt":"latent universal physics transformer","gino":"graph lifting, latent FNO, and point projection","domino":"decomposable multiscale iterative neural operator","p3d":"scalable 3D global-context surrogate","aerotransformer":"pretrained transformer for 3D aerodynamics","inc":"equation-level indirect neural corrector","pict":"differentiable GPU multi-block PISO solver","diffsph":"differentiable smoothed-particle hydrodynamics","np_newton":"neural-operator preconditioned Newton solver","geometry_preconditioner":"geometry-aware neural iterative preconditioner"};
const REF = {"pibert":"Chakraborty, Pan & Chen, 2026","fno":"Li et al., ICLR 2021","deeponet":"Lu et al., Nature Machine Intelligence 2021","transolver":"Wu et al., ICML 2024","upt":"Alkin et al., NeurIPS 2024","gino":"Li et al., NeurIPS 2023","inc":"Wei et al., NeurIPS 2025","pict":"Franz et al., JCP 2025","diffsph":"Winchenbach & Thuerey, JCP 2026"};
const REPO = {"pibert":"https://github.com/Samsomyajit/pibert","fno":"https://github.com/neuraloperator/neuraloperator","geo_fno":"https://github.com/neuraloperator/Geo-FNO","transolver":"https://github.com/thuml/Neural-Solver-Library","inc":"https://github.com/tum-pbs/INC"};

export const models = MODEL_ROWS.map(([id,name,categories]) => {
  const geometry = GEOMETRY.has(id);
  const threeD = THREED.has(id) || !["pinnsformer","riemannonet"].includes(id);
  const accel = ACCEL.has(id);
  const particle = PARTICLE.has(id);
  return {
    id,name,categories,
    architecture: ARCH[id] || `${name} architecture`,
    tasks: accel ? ["acceleration","corrector","preconditioner"] : ["surrogate","forecasting"],
    physics: particle ? ["particle","granular"] : ["general_pde","fluid_dynamics"],
    mesh_types: particle ? ["particle"] : (geometry ? ["unstructured","point_cloud","structured"] : ["structured","meshfree"]),
    geometry_modes: geometry ? ["varying","parameterized"] : ["fixed","parameterized"],
    temporal_modes: (LONG.has(id) || accel) ? ["autoregressive","unsteady","steady"] : ["steady","unsteady"],
    dimensions: threeD ? [1,2,3] : [1,2],
    strengths:["registered in the uniform NAVIER-CFD model taxonomy"],
    limitations:["verify claims on the selected CFD benchmark and discretization"],
    reference: REF[id] || name,
    repository: REPO[id] || null,
    integration: REPO[id] ? "external" : "metadata",
    min_memory_gb: ["gino","domino","p3d","aerotransformer","neuraldem"].includes(id) ? 24 : 8,
    framework:"pytorch",
    tags:[
      geometry ? "mesh_transfer" : "",
      (LONG.has(id) || accel) ? "long_rollout" : "",
      UQ.has(id) ? "uncertainty" : "",
      accel ? "conservative" : "",
      threeD ? "3d" : ""
    ].filter(Boolean)
  };
});
