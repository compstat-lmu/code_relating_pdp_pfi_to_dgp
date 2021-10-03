#' Compute Importance for mlr3 resampling object
#' 
#' Currently hard coded for MSE
#'
#' @param rs An mlr3 resampling object
#' @param agg Returns the averaged importance per feature if TRUE, otherwise one per fold
#' @param dat Dataset with which to compute the importance. Uses complement of fold if dat=NULL
#' @return Data.frame with columns feature and importance. multiple entries per feature if agg=FALSE
rsmp_imp = function(rs, dat = NULL, nperm = 30, alpha = 0.05){
  nfolds = length(rs$learners)
  t_alpha = qt(1 - alpha / 2, df = nfolds - 1)
  target = rs$task$target_names
  pfis = lapply(1:nfolds, function(k){
    test_set_index = rs$resampling$test_set(k)
    if (is.null(dat)) {
      test_set = rs$task$data(rows = test_set_index)
    } else {
      test_set = dat
    }
    test_task = TaskRegr$new("1",  test_set, target = target)
    model = rs$learners[[k]]
    res = pfi_var(model, test_task, nperm = nperm, alpha = alpha)
    res$fold = k
    res
  })
  pfis = rbindlist(pfis)
  n2 = length(rs$resampling$test_set(1))
  n1 = length(rs$resampling$train_set(1))
  res = pfis %>%
    dplyr::group_by(feature) %>%
    dplyr::summarize(var3 = (1 / nfolds + n2/n1) * var(pfi),
                     pfi = mean(pfi),
                     lower = pfi - t_alpha * sqrt(var3),
                     upper = pfi + t_alpha * sqrt(var3))
     
  res
}

