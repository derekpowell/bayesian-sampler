library(tidyverse)

Rcpp::cppFunction('NumericVector rcpp_clip( NumericVector x, double a, double b){
    return clamp( a, x, b ) ;
}')
sample_corr_binary <- function(N, p, autocor_p){
  samps <- rbernoulli(1, p)

  for (i in 2:N){
      samps <- c(samps, ifelse(rbernoulli(1, autocor_p), samps[length(samps)], rbernoulli(1,p)))
  }
  return(as.numeric(samps))
}

sample_corr_binary_2init <- function(N, p, autocor_p){
  samps0 <- 0
  samps1 <- 1

  for (i in 1:N){
    if (rbernoulli(1, .5)){
      samps0 <- c(samps0, ifelse(rbernoulli(1, autocor_p), samps0[length(samps0)], rbernoulli(1,p)))
    } else {
      samps1 <- c(samps1, ifelse(rbernoulli(1, autocor_p), samps1[length(samps1)], rbernoulli(1,p)))
    }

    samps <- c(samps0[1:length(samps0)], samps1[1:length(samps1)])

  }
  return(samps)
}

est_weight <- function(p, p_auto, n=100){
  x <- sample_corr_binary(n, p, p_auto)
  weight <- rcpp_clip(coda::effectiveSize(x)[[1]]/n,.005,1)

  return(weight)
}


rapprox <- function(n, p, N_eff){
  # create a mixture distribution ?
  x <- sample_corr_binary(100, p, .7)
  weight <- rcpp_clip(coda::effectiveSize(x)[[1]]/100,.005,1)

  # N <- round(N_eff/weight)
  p1 <- p^N_eff
  p0 <- (1-p)^N_eff

  ones <- rep(1, round(p1*n))
  zeroes <- rep(0, round(p0*n))

  output <- rbeta(n-round(p1*n)-round(p0*n), p*N_eff, (1-p)*N_eff)

  return(c(ones, zeroes, output))
}

sample_autocorrelated_ess <- function(p, N_eff){
  ## Draw autocorrelated samples to achieve effective sample size
  x <- sample_corr_binary(100, p, .7)
  weight <- rcpp_clip(coda::effectiveSize(x)[[1]]/100,.005,1)
  # weight <- rbeta(1, 10, 50) # don't need noise but would be reasonable
  N <- round(N_eff/weight)  # N = ess/w
  weight <- N_eff/N

  Sa <- sample_corr_binary(N, p, .7)
  # resamp <- rbernoulli(N, p = .1) # d = .05, misraeding not autocorrelated
  # Sa <- ifelse(resamp, !Sa, Sa)
  Sa <- sum(Sa*weight)#/N_eff

  return(Sa)
}

sample_autocorrelated_ess_init2 <- function(p, N_eff, p_auto=.7){
  ## Draw autocorrelated samples to achieve effective sample size
  x <- sample_corr_binary_2init(500, rcpp_clip(p,.02,1), p_auto) # estimates get wonky if you let p too small
  weight <- rcpp_clip(coda::effectiveSize(x)[[1]]/502,.01,1)
  N <- round(N_eff/weight)  # N = ess/w
  # weight <- rbeta(N, weight*10, (1-weight)*10) # don't need noise but would be reasonable

  Sa <- sample_corr_binary_2init(N-2, p, p_auto)
  Sa <- sum(Sa*weight)

  return(Sa)
}


rBinEss <- function(n, p, N_eff){
  return(replicate(n, sample_autocorrelated_ess(p,N_eff)/N_eff))
}

rBinEssInit <- function(n, p, N_eff, p_auto=.7){
  return(replicate(n, sample_autocorrelated_ess_init2(p,N_eff,p_auto)/N_eff))
}
