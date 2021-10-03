#' Compute Model-PFI along with variance estimators and CIs 
#'
#' @param model The mlr3::Learner for which to compute the PFIs
#' @param task The mlr3::Task to estimate the PFIs and variances
#' @param nperm Number of permutations
#' @param alpha Confidence level for confidence intervals
#' @param row_ids vector with ids to compute the importance. If not provided,
#'          uses all data points in task
pfi_var = function(model, task, nperm = 30, alpha = 0.05, row_ids = NULL){
  checkmate::assert_class(model, "Learner")
  checkmate::assert_class(task, "Task")
  checkmate::assert_numeric(nperm, lower = 3, max.len = 1, any.missing = FALSE)
  checkmate::assert_numeric(alpha, lower = 0, upper = 1, max.len = 1, any.missing = FALSE) 
  checkmate::assert_numeric(row_ids, lower = 1, upper = task$nrow,
                            max.len = task$nrow, any.missing = FALSE, null.ok = TRUE)
  if (is.null(row_ids)) row_ids = task$row_ids

  # Extracting information from task
  dat = task$data(row_ids)
  ntest = nrow(dat)
  t_alpha = qt(1 - alpha / 2, df = ntest) 
  y = dat[[task$target_names]]
  features = task$feature_names
  # Computing the original instance-wise losses for the given task
  predicted = model$predict(task, row_ids = row_ids)$response
  l_orig = (y - predicted)^2
  # Computing instance-wise permutation losses per feature and per permutation
  losses = lapply(features, function(fname){
    dat2 = dat
    res = lapply(1:nperm, function(k){
      dat2[,fname] = sample(dat2[[fname]])
      task2 = TaskRegr$new("s", dat2, task$target_names)
      predicted2 = model$predict(task2)$response
      loss_shuffle = (y - predicted2)^2
      data.frame(l_shuffle = loss_shuffle, l_orig = l_orig, shuffle_index = k, id = 1:length(loss_shuffle))
    })
    res = rbindlist(res)
    res$feature = fname
    res
  })
  losses = data.frame(rbindlist(losses))
  # Computing second term of the variance sum
  variance = losses %>%
    dplyr::group_by(feature, id) %>%
    dplyr::summarize(l_diff = mean(l_shuffle - l_orig)) %>%
    dplyr::group_by(feature) %>%
    dplyr::summarize(vari = (1/ntest) *  var(l_diff))
  
  # Computing feature importance
  pfis = losses %>%
    dplyr::group_by(feature) %>%
    dplyr::summarize(pfi = mean(l_shuffle - l_orig))
   
  variance = merge(variance, pfis) %>%
    dplyr::mutate(se = sqrt(vari),
                  lower = pfi - t_alpha * se,
                  upper = pfi + t_alpha * se)
  variance
}
