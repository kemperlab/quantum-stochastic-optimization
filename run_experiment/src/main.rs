use color_eyre::Result;
use serde::{Deserialize, Serialize};
use serde_json::to_string_pretty;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizationRuns {
    seed: Seed,
    shots: usize,
    resampling: ResamplingParameters,
    optimizer: OptimizerParameters,
    problem: ProblemParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum Seed {
    Random,
    RunNumber,
    Seed { seed: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResamplingParameters {
    epsilon: f64,
    hamiltonians: usize,
    split_shots: bool,
    resample_single: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum OptimizerParameters {
    Adam(AdamParameters),
    Spsa(SpsaParameters),
    TrustRegion(TrustRegionParameters),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum ProblemParameters {
    FeatureSelection(FeatureSelectionProblem),
    TightBinding(TightBindingProblem),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdamParameters {
    alpha: f64,
    beta_1: f64,
    beta_2: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpsaParameters {
    step_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrustRegionParameters {
    delta_0: f64,
    rho: f64,
    gamma_1: f64,
    gamma_2: f64,
    mu: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureSelectionProblem {
    k_real: usize,
    k_fake: usize,
    k_redundant: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TightBindingProblem {
    n_atoms: usize,
    alpha: Distribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
enum Distribution {
    Normal { mu: f64, sigma: f64 },
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let args = OptimizationRuns {
        seed: Seed::RunNumber,
        shots: 1024,
        resampling: ResamplingParameters {
            epsilon: 0.1,
            hamiltonians: 1,
            split_shots: false,
            resample_single: true,
        },
        optimizer: OptimizerParameters::Adam(AdamParameters {
            alpha: 0.01,
            beta_1: 0.9,
            beta_2: 0.999,
        }),
        problem: ProblemParameters::TightBinding(TightBindingProblem {
            n_atoms: 5,
            alpha: Distribution::Normal {
                mu: 10.,
                sigma: 1.5,
            },
        }),
    };

    println!("{}", to_string_pretty(&args)?);

    Ok(())
}
