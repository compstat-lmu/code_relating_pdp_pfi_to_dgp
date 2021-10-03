# Resampled PDP
# Has to be initialized with resample object
# Computes for each model the PDP and provids a predict function that is the average of PDPs
RsmpPdp <- R6::R6Class("rsmp-pdp",
  public = list(
    # list of PDP objects
    pdps = NULL,
    # Resample object
    rs = NULL,
    # Selected feature
    feature = NULL,
    # The variance per PDP
    variances = NULL,
    # Name of the target
    target = NULL,
    #' @description Create PDP based on mlr3::Resampling object
    #' @param rs mlr3 resampling object
    #' @param feature Feature name
    #' @param dat The data with which to compute the PDP. If empty, the resampling object decides.
    initialize = function(rs, feature, dat = NULL, alpha = 0.05, grid = NULL){
      self$feature = feature
      self$rs = rs
      self$target = rs$task$target_names
      if(is.null(grid)){
        grid = seq(from = min(rs$task$data()[[feature]]),
                   to   = max(rs$task$data()[[feature]]),
                   length.out = 30)
      }
      # The number of models
      nfolds = length(rs$learners)
      self$pdps = variances = list()
      for (k in 1:nfolds) {
        if (is.null(dat)) {
          test_set_index = rs$resampling$test_set(k)
          test_set = rs$task$data(rows = test_set_index)
        } else {
          test_set = dat
        }
        model = rs$learners[[k]]
        if(inherits(model, "AutoTuner")){
          pred = Predictor$new(model$learner, data = data.frame(test_set), y = self$target)
        } else {
          # There seems to be an issue with retrieving Learners from mlr3::ResampleResult
          # But only when it is not an AutoTuner
          # circumvention in two steps: 1. set task in Learner. 2. use custom predict function
          model2 = model
          model2$state$train_task = rs$task
          predfun = function(model, newdata) {model2$predict(TaskRegr$new("t", newdata, self$target))$response}
          pred = Predictor$new(data = data.frame(test_set), predict.function = predfun)
        }
        # A check if correctly  working now
        #check1 = pred$predict(test_set[c(3,1,2),])[[1]]
        #check2 = model$predict(TaskRegr$new("t", test_set[c(3,1,2),], self$target))$response
        #names(check2) = NULL
        #checkmate::assert_true(all.equal(check1, check2))
        #cnames = setdiff(colnames(test_set), self$target)
        #checkmate::assert_true(all.equal(data.frame(pred$data$X)[,cnames], data.frame(test_set)[,cnames]))
        # End of check

        self$pdps[[k]] = FeatureEffect$new(pred, feature = feature, method = "pdp", grid.points = grid)
        variances[[k]] = pdp_ci(pred, feature = feature, grid.points = grid)
      }
      variances = rbindlist(variances)
      # In case of bootstrapping, we need the mean
      n2 = mean(unlist(lapply(1:nfolds, function(k) length(unique(rs$resampling$test_set(k))))))
      n1 = mean(unlist(lapply(1:nfolds, function(k) length(unique(rs$resampling$train_set(k))))))
      t_alpha = qt(1 - alpha / 2, df = nfolds - 1)
      self$variances = variances %>%
        dplyr::group_by(across(feature)) %>%
        dplyr::summarize(var = (1/nfolds + n2/n1) * var(pdp),
                         pdp = mean(pdp),
                         lower = pdp - t_alpha * sqrt(var),
                         upper = pdp + t_alpha * sqrt(var))
    },
    predict = function(x, extrapolate = FALSE){
      predictions = lapply(self$pdps, function(pdp){
        pdp$predict(x, extrapolate = extrapolate)
      })
      rowMeans(data.frame(predictions))
    }
  )
)


