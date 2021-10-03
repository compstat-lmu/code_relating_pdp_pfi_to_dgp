#' Compute the PDP for a model plus vars
#'
#' @param pred an iml::Predictor
#' @param feature Name of the feature for the PDP
#' @param grid.points, optional grid.points for the PDP
#' @return data.frame with PDP, variance and CI estimators
pdp_ci = function(pred, feature, grid.points = NULL){
  ice = FeatureEffect$new(pred, method = "ice", feature = feature, grid.points = grid.points)
  pdp = ice$results %>%
    dplyr::group_by_at(feature) %>%
    dplyr::summarize(pdp = mean(.value),
                     n = n(),
                     pdp_est_var = var(.value) / n,
                     .groups = "drop")
  colnames(pdp)[1] = feature
  pdp
}


