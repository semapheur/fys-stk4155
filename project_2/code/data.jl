using CSV
using DataFrames
using Downloads

function fetch_wisconsin_breast_cancer_data()::Tuple{Matrix{Float64},Vector{Float64}}
  feature_names = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
  ]

  url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
  df = CSV.File(Downloads.download(url)) |> DataFrame

  rename!(df, vcat(:id, :diagnosis, [Symbol(name) for name in feature_names]))

  df.diagnosis = df.diagnosis .== "M"
  X = Matrix(df[:, feature_names])
  y = Float64.(df.diagnosis)

  return X, y
end