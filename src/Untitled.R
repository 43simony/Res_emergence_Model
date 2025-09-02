library(ggplot2)
library(dplyr)
library(patchwork)   
library(scales)
# ---------------------------
# Core multi-type PGF 
# ---------------------------
# f_gen returns the one-generation PGF component for parent type `idx` (1..4)
f_gen <- function(S, idx, b, d, mu) {
  order <- list(
    c(1, 2, 3, 4), # WT
    c(2, 1, 4, 3), # M1
    c(3, 4, 1, 2), # M2
    c(4, 3, 2, 1)  # M12
  )[[idx]]
  # Treat inputs b,d as *rates* and convert to per-generation probs (b+d > 0)
  if ((b + d) <= 0) stop("b + d must be > 0")
  d_prob <- d / (b + d)
  b_prob <- b / (b + d)
  
  d_prob +
    b_prob * (1 - 2*mu + mu^2) * S[order[1]]^2 +
    b_prob * mu * (1 - mu) * S[order[2]] * S[order[1]] +
    b_prob * mu * (1 - mu) * S[order[3]] * S[order[1]] +
    b_prob * (mu^2) * S[order[4]] * S[order[1]]
}

# Vectorized mapping for all 4 types
F_map <- function(S, b, d, mu) {
  sapply(1:4, function(i) f_gen(S, i, b, d, mu))
}

# ---------------------------
# Iterate PGF along axis with z placed on coordinate `type_idx`
# ---------------------------
iterate_F_axis <- function(s, N_gen, b, d, mu) {
  S = s
  if (N_gen == 0) return(S)
  for (k in seq_len(N_gen)) S <- F_map(S, b, d, mu)
  S
}

# Full-population PGF evaluated when only coordinate `type_idx` is variable (others = 1)
G_total <- function(z, N_gen, n0, b, d, mu) {
  S <- iterate_F_axis(s = z, N_gen, b, d, mu)
  terms <- mapply(function(Si, ni) if (ni == 0) 1+0i else Si^ni, S, n0)
  prod(terms)
}

# ---------------------------
# FFT inversion: recover PMF for chosen type
# ---------------------------

pmf_type_fft <- function(N_gen, n0, b, d, mu, K = 4096L, type_idx = 4, verbose = FALSE) {
  stopifnot(length(n0) == 4L)
  k <- 0:(K-1)
  z_k <- exp(2 * pi * 1i * k / K)
  
  # Evaluate PGF on roots of unity in log-space to avoid underflow
  Gk <- complex(length = K)
  for (idx in seq_along(z_k)) {
    z <- complex(real = 1, length = 4)
    z[type_idx] <- z_k[idx]
    S <- iterate_F_axis(s = z, N_gen, b, d, mu)  # complex vector length 4
    
    # If any S component is exactly 0, log(S) -> -Inf, so handle:
    # For any S_j == 0 and n0[j] > 0, the contribution is zero (product is zero).
    # Compute logG = sum_j n0[j] * log(S_j). Use complex log.
    if (any(Mod(S) == 0 & n0 != 0)) {
      Gk[idx] <- 0+0i
    } else {
      logS <- log(S)              # complex log
      logG  <- sum(as.numeric(n0) * logS)
      Gk[idx] <- exp(logG)       # complex value
    }
  }
  
  if (verbose) {
    cat("Gk magnitude summary:\n")
    print(summary(Mod(Gk)))
    cat("any NaN? ", any(is.nan(Re(Gk)) | is.nan(Im(Gk))), "\n")
    cat("any infinite? ", any(is.infinite(Re(Gk)) | is.infinite(Im(Gk))), "\n")
  }
  
  # inverse FFT -> coefficients and normalization
  ## coeffs <- fft(Gk, inverse = TRUE) / K
  coeffs <- fft(Gk, inverse = FALSE) 
  #pmf <- pmax(0, Re(coeffs))    # clip tiny negative noise
  pmf <- Re(coeffs)
  total <- sum(pmf)
  if (total <= 0) {
    warning("pmf sum <= 0 after clipping; returning tiny-noise vector. Check Gk diagnostics.")
    return(rep(0, K))
  }
  pmf <- pmf / total
  pmf
}


pmf_type_MVfft <- function(N_gen, n0, b, d, mu, K = 128L, type_idx = 4, verbose = FALSE) {
  stopifnot(length(n0) == 4L)
  type_idx <- as.integer(type_idx)
  d_out <- length(type_idx)  # dimension of joint pmf
  
  # grid sizes: same K in each dimension for simplicity
  k_list <- replicate(d_out, 0:(K-1), simplify = FALSE)
  z_grids <- lapply(k_list, function(k) exp(2 * pi * 1i * k / K))
  
  # build meshgrid of evaluation points
  grid <- expand.grid(k_list)
  n_eval <- nrow(grid)
  Gk <- complex(length = n_eval)
  
  if (verbose) cat("Evaluating PGF on", n_eval, "grid points...\n")
  
  for (idx in seq_len(n_eval)) {
    z_vec <- rep(1+0i, 4)  # 4D input to PGF
    z_vec[type_idx] <- mapply(function(k, dim) z_grids[[dim]][k+1],
                              k = as.numeric(grid[idx, ]), dim = seq_along(type_idx))
    
    S <- iterate_F_axis(s = z_vec, N_gen, b, d, mu)  # now needs to accept full z_vec
    
    if (any(Mod(S) == 0 & n0 != 0)) {
      Gk[idx] <- 0+0i
    } else {
      logS <- log(S)
      logG <- sum(as.numeric(n0) * logS)
      Gk[idx] <- exp(logG)
    }
  }
  
  # Reshape into array for multi-FFT
  Gk_array <- array(Gk, dim = rep(K, d_out))
  
  # Apply multidimensional inverse FFT
  coeffs <- fft(Gk_array, inverse = FALSE)  # in R, multi-d FFT works on arrays
  pmf <- Re(coeffs)
  
  # Normalize
  total <- sum(pmf)
  if (total <= 0) {
    warning("pmf sum <= 0 after FFT; returning zeros")
    return(array(0, dim = rep(K, d_out)))
  }
  pmf <- pmf / total
  return(pmf)
}


# ---------------------------
# Mean matrix (Jacobian) and K heuristic
# ---------------------------
mean_matrix <- function(b, d, mu) {
  eps <- 1e-8
  base <- c(1, 1, 1, 1)
  F0 <- F_map(base, b, d, mu)
  M <- matrix(0, 4, 4)
  for (j in 1:4) {
    Sj <- base; Sj[j] <- Sj[j] + eps
    Fj <- F_map(Sj, b, d, mu)
    M[, j] <- Re((Fj - F0) / eps)
  }
  M
}

auto_K <- function(N_gen, n0, b, d, mu, L = 8, type_idx = 4) {
  M <- mean_matrix(b, d, mu)
  Ez <- as.numeric(n0)
  for (t in seq_len(N_gen)) Ez <- as.numeric(Ez %*% M)
  mean_i <- Ez[type_idx]
  sd_i   <- sqrt(max(1, mean_i + 0.25 * mean_i^2))
  2^ceiling(log2(max(32, ceiling(mean_i + L * sd_i + 10))))
}

# ---------------------------
# Wrapper for calling your C++ binary (unchanged pattern)
# Modify path/exe as needed for your environment
# ---------------------------
multiType_BP <- function(n_reps, parameters, exe_path = "./multiType_BP.exe", run_dir = NULL) {
  parvec <- c(format(n_reps, scientific = FALSE),
              parameters$N_WT, parameters$N_M1, parameters$N_M2, parameters$N_M12,
              parameters$b_rate, parameters$d_rate, parameters$mu_prob,
              parameters$T_max, parameters$data_out, parameters$batchname)
  strvec <- format(parvec, digits = 5)
  # Optionally change working directory for the run
  oldwd <- getwd()
  if (!is.null(run_dir)) setwd(run_dir)
  res <- system2(exe_path, args = strvec, stdout = TRUE)
  out <- read.table(text = res, header = TRUE, sep = ";", check.names = FALSE) %>%
    mutate_all(as.numeric)
  if (!is.null(run_dir)) setwd(oldwd)
  out
}

# ---------------------------
# Compare & plot: simulation histogram vs analytic PMF
# options:
#    - type_idx: 1..4 (WT, M1, M2, M12)
#    - Gen_val: generation to inspect
# ---------------------------
plot_compare_pdf <- function(out_df, pars, type_idx = 4, Gen_val = 0,
                             L = 8, autoK = TRUE, K = NULL) {
  stopifnot(type_idx %in% 1:4)
  type_names <- c("n_WT", "n_M1", "n_M2", "n_M12")
  colname <- type_names[type_idx]
  
  # Extract simulation counts at generation T
  sim_df <- out_df %>%
    filter(Gen == Gen_val) %>%
    select(all_of(colname)) %>%
    rename(count = all_of(colname))
  
  # Empirical histogram
  sim_hist <- sim_df %>%
    count(count, name = "freq") %>%
    mutate(prob = freq / sum(freq))
  
  # Prepare analytic PMF
  ##b <- pars$b_rate / (pars$b_rate+pars$d_rate)
  ##d <- pars$d_rate / (pars$b_rate + pars$d_rate) 
  b <- pars$b_rate
  d <- pars$d_rate
  mu <- pars$mu_prob
  n0 <- c(pars$N_WT, pars$N_M1, pars$N_M2, pars$N_M12)
  
  if (autoK) {
    K <- auto_K(N_gen = Gen_val, n0, b, d, mu, L = L, type_idx = type_idx)
  } else {
    if (is.null(K)) stop("K must be provided when autoK = FALSE")
  }
  
  pmf <- pmf_type_fft(N_gen = Gen_val, n0, b, d, mu, K = K, type_idx = type_idx, verbose = T)
  ana_df <- data.frame(n = 0:(length(pmf) - 1), prob = pmf)
  
  # Determine xmax robustly (if cumsum never reaches threshold use full range)
  threshold_idx <- which(cumsum(pmf) >= 0.999)
  xmax_idx <- if (length(threshold_idx) > 0) threshold_idx[1] else length(pmf)
  xmax <- max(max(sim_hist$count, na.rm = TRUE), xmax_idx)
  
  # Filter both dataframes: range, drop zeros, modulus filter
  sim_hist <- sim_hist %>% filter(count <= xmax, prob > 0)
  
  ana_df <- ana_df %>% filter(n <= xmax, prob > 0)
  
  # Plot
  p <- ggplot() +
    geom_col(data = sim_hist, aes(x = count, y = prob), fill = "skyblue", alpha = 0.6, width = 1) +
    geom_line(data = ana_df, aes(x = n, y = prob), color = "red", size = 1.2) +
    labs(title = sprintf("Distribution of %s at generation T=%d", type_names[type_idx], Gen_val),
         x = sprintf("%s count", type_names[type_idx]),
         y = "Probability") +
    theme_minimal(base_size = 14)
  print(p)
  
  ## return output data
  return(list(sim_hist, ana_df))
  
}

plot_compare_pdf_v2 <- function(out_df, pars, Gen_val = 0,
                             L = 8, autoK = TRUE, K = NULL) {
  
  type_names <- c("n_WT", "n_M1", "n_M2", "n_M12")
  
  all_data <- list()
  plots <- list()
  
  for (type_idx in 1:4) {
    colname <- type_names[type_idx]
    
    # Extract simulation counts at generation T
    sim_df <- out_df %>%
      filter(Gen == Gen_val) %>%
      select(all_of(colname)) %>%
      rename(count = all_of(colname))
    
    # Empirical histogram
    sim_hist <- sim_df %>%
      count(count, name = "freq") %>%
      mutate(prob = freq / sum(freq),
             type = type_names[type_idx])
    
    # Analytic PMF
    b <- pars$b_rate
    d <- pars$d_rate
    mu <- pars$mu_prob
    n0 <- c(pars$N_WT, pars$N_M1, pars$N_M2, pars$N_M12)
    
    if (autoK) {
      K_val <- auto_K(N_gen = Gen_val, n0, b, d, mu, L = L, type_idx = type_idx)
    } else {
      if (is.null(K)) stop("K must be provided when autoK = FALSE")
      K_val <- K
    }
    
    pmf <- pmf_type_fft(N_gen = Gen_val, n0, b, d, mu, K = K_val, type_idx = type_idx)
    ana_df <- data.frame(n = 0:(length(pmf) - 1), prob = pmf, type = type_names[type_idx])
    
    # xmax cutoff
    threshold_idx <- which(cumsum(pmf) >= 0.999)
    xmax_idx <- if (length(threshold_idx) > 0) threshold_idx[1] else length(pmf)
    xmax <- max(max(sim_hist$count, na.rm = TRUE), xmax_idx)
    
    sim_hist <- sim_hist %>% filter(count <= xmax, prob > 0)
    ana_df   <- ana_df %>% filter(n <= xmax, prob > 0)
    
    # Plot for this type
    p <- ggplot() +
      geom_col(data = sim_hist, aes(x = count, y = prob), 
               fill = "skyblue", alpha = 0.6, width = 1) +
      geom_line(data = ana_df, aes(x = n, y = prob), 
                color = "red", size = 1.2) +
      labs(title = sprintf("%s (T=%d)", type_names[type_idx], Gen_val),
           x = "Count", y = "Probability") +
      theme_minimal(base_size = 14)
    
    print(p)
    
    plots[[type_idx]] <- p
    all_data[[type_idx]] <- list(sim_hist = sim_hist, ana_df = ana_df)
  }
  
  # Arrange plots in 2x2 grid
  final_plot <- (plots[[1]] | plots[[2]]) / (plots[[3]] | plots[[4]])
  print(final_plot)
  
  # Option 1: return list of dfs
  return(all_data)
  
  # Option 2 (instead of above): bind into one dataframe
  # combined_df <- bind_rows(lapply(all_data, function(x) bind_rows(x$sim_hist, x$ana_df)))
  # return(combined_df)
}

plot_compare_pdf_MV <- function(out_df, pars, type_idx = c(), Gen_val = 0,
                             L = 8, autoK = TRUE, K = NULL) {
  stopifnot(all(type_idx %in% 1:4))
  type_names <- c("n_WT", "n_M1", "n_M2", "n_M12")
  
  # --- Simulation data ---
  sim_df <- out_df %>%
    filter(Gen == Gen_val) %>%
    select(all_of(type_names[type_idx]))
  
  # Collapse into counts if multivariate
  sim_counts <- sim_df %>%
    count(across(everything()), name = "freq") %>%
    mutate(prob = freq / sum(freq))
  
  # --- Parameters ---
  b <- pars$b_rate
  d <- pars$d_rate
  mu <- pars$mu_prob
  n0 <- c(pars$N_WT, pars$N_M1, pars$N_M2, pars$N_M12)
  
  if (autoK) {
    K <- auto_K(N_gen = Gen_val, n0, b, d, mu, L = L, type_idx = type_idx)
  } else {
    if (is.null(K)) stop("K must be provided when autoK = FALSE")
  }
  
  # --- Analytic PMF via FFT ---
  pmf <- pmf_type_MVfft(
    N_gen = Gen_val, n0, b, d, mu,
    K = K, type_idx = type_idx, verbose = TRUE
  )
  
  # pmf now could be vector (1D), matrix (2D), or array (kD)
  
  # --- Build analytic dataframe ---
  if (length(type_idx) == 1) {
    ana_df <- data.frame(n = 0:(length(pmf) - 1), prob = pmf)
  } else {
    dims <- dim(pmf)
    idx_grid <- as.data.frame(do.call(expand.grid, lapply(dims, function(m) 0:(m-1))))
    colnames(idx_grid) <- type_names[type_idx]
    ana_df <- cbind(idx_grid, prob = as.vector(pmf))
  }
  
  # --- Plotting ---
  if (length(type_idx) == 1) {
    # Univariate plot
    p <- ggplot() +
      geom_col(data = sim_counts, aes(x = !!sym(type_names[type_idx]), y = prob),
               fill = "skyblue", alpha = 0.6, width = 1) +
      geom_line(data = ana_df, aes(x = n, y = prob), color = "red", size = 1.2) +
      labs(title = sprintf("Distribution of %s at generation T=%d",
                           type_names[type_idx], Gen_val),
           x = sprintf("%s count", type_names[type_idx]),
           y = "Probability") +
      theme_minimal(base_size = 14)
    print(p)
  } else if (length(type_idx) == 2) {
    # Bivariate heatmap
    
    p <- ggplot() +
      geom_tile(data = sim_counts, aes(x = !!sym(type_names[type_idx[1]]),
                                       y = !!sym(type_names[type_idx[2]]), 
                                       fill = prob)) +
      geom_contour(data = ana_df, aes(x = !!sym(type_names[type_idx[1]]),
                                      y = !!sym(type_names[type_idx[2]]),
                                      z = pmin(prob, cap_val)), 
                   bins = 7, color = '#00FFFF', alpha = 0.8) +
      scale_fill_viridis_c(
        option = "plasma",
        limits = c(0, cap_val),       # cap the upper end
        oob = squish                  # squash anything above cap_val
      ) +
      labs(title = sprintf("Joint distribution of (%s,%s) at generation T=%d",
                           type_names[type_idx[1]], type_names[type_idx[2]], Gen_val),
           x = type_names[type_idx[1]],
           y = type_names[type_idx[2]]) +
      theme_minimal(base_size = 14)
    
    print(p)
    
    cap_val <- quantile(ana_df$prob, probs = 0.99, na.rm = TRUE)
    x_range <- quantile(sim_counts[[ type_names[type_idx[1]] ]], c(0, 0.9), na.rm = TRUE)
    x_range[2] <- ceiling(x_range[2] / 5) * 5
    y_range <- quantile(sim_counts[[ type_names[type_idx[2]] ]], c(0, 0.9), na.rm = TRUE)
    y_range[2] <- ceiling(y_range[2] / 5) * 5
    
    p <- ggplot() +
      geom_tile(data = sim_counts, aes(x = !!sym(type_names[type_idx[1]]),
                                       y = !!sym(type_names[type_idx[2]]), 
                                       fill = prob)) +
      geom_contour(data = ana_df, aes(x = !!sym(type_names[type_idx[1]]),
                                      y = !!sym(type_names[type_idx[2]]),
                                      z = pmin(prob, cap_val)), 
                   bins = 7, color = '#00FFFF', alpha = 0.8) +
      scale_fill_viridis_c(
        option = "plasma",
        limits = c(0, cap_val),       # cap the upper end
        oob = squish                  # squash anything above cap_val
      ) +
      labs(title = sprintf("Joint distribution of (%s,%s) at generation T=%d",
                           type_names[type_idx[1]], type_names[type_idx[2]], Gen_val),
           x = type_names[type_idx[1]],
           y = type_names[type_idx[2]]) +
      scale_x_continuous(limits = x_range, expand = c(0, 0)) +
      scale_y_continuous(limits = y_range, expand = c(0, 0)) +
      theme_minimal(base_size = 14)
    
    print(p)
  } else {
    message("Joint distributions for 3+ types are returned as dataframes but not plotted.")
  }
  
  ## add here
  marginal_dfs <- list()
  
  for (var in type_names[type_idx]) {
    # Analytic marginal: sum over all other variables
    ana_marg <- ana_df %>%
      group_by(!!sym(var)) %>%
      summarise(prob = sum(prob, na.rm = TRUE), .groups = "drop") %>%
      mutate(source = "Analytic")
    
    # Simulation marginal: sum over all other variables
    sim_marg <- sim_counts %>%
      group_by(!!sym(var)) %>%
      summarise(prob = sum(prob, na.rm = TRUE), .groups = "drop") %>%
      mutate(source = "Simulated")
    
    # Combine into one df
    marg_df <- bind_rows(ana_marg, sim_marg) %>%
      mutate(variable = var)
    
    marginal_dfs[[var]] <- marg_df
    
    # --- Plot each marginal ---
    p_marg <- ggplot(marg_df, aes(x = !!sym(var), y = prob)) +
      geom_col(data = subset(marg_df, source == "Simulated"), fill = "skyblue", alpha = 0.6, width = 1) +
      geom_line(size = 1.2, data = subset(marg_df, source == "Analytic"), color = "red") +
      labs(title = sprintf("Marginal distribution of %s", var),
           x = sprintf("%s count", var),
           y = "Probability") +
      theme_minimal(base_size = 14)
    
    print(p_marg)
  }
  
  # Collect all marginals into one dataframe for later use
  marg <- bind_rows(marginal_dfs, .id = "marginal_var")
  
  ## merge data
  ana_df <- ana_df %>%
    left_join(sim_counts %>% select(-freq), 
              by = colnames(sim_counts)[!colnames(sim_counts) %in% c("freq","prob")]) %>%
    rename(sim_prob = prob.y,
           prob = prob.x)
  
  return(list(ana = ana_df, marg = marg))
}



##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


pars <- data.frame(N_WT = 1, N_M1 = 0, N_M2 = 0, N_M12 = 0,
                   b_rate = 2, d_rate = 1, mu_prob = 1e-3,
                   T_max = 50, data_out = 2, batchname = 'test')

# Run simulation (make sure exe_path/run_dir are correct for your setup)
tic = Sys.time()
out <- multiType_BP(n_reps = 100000, parameters = pars, exe_path = "./multiType_BP_fast.exe", run_dir = "~/Desktop/Repos/Res_emergence_Model/src")
toc = Sys.time()
toc - tic

gen_dist <- plot_compare_pdf(out_df = out, pars, type_idx = 1, Gen_val = 7, autoK = F, K = 2^6)

gen_dist <- plot_compare_pdf_v2(out_df = out, pars, Gen_val = 7, autoK = F, K = 2^6)

gen_dist <- plot_compare_pdf_MV(out_df = out, pars, type_idx = c(3,4), Gen_val = 50, autoK = F, K = 2^7)








################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################
################################################################################################################################################################################

I0 = data.frame(WT = 1, M1 = 0, M2 = 0, M12 = 0)
nrep = 1000
B = c(2,2,2,2)
D_0 = c(1, 1, 1, 1)
D_T = c(3, 3, 2.5, 1)
mu = 1e-3
mu_0 = (1-mu)^2
mu_1 = mu*(1-mu)
mu_2 = mu^2

mu_matrix <- list(
  c(mu_0, mu_1, mu_1, mu_2), # WT
  c(mu_1, mu_0, mu_2, mu_1), # M1
  c(mu_1, mu_2, mu_0, mu_1), # M2
  c(mu_2, mu_1, mu_1, mu_0)  # M12
)

N_gen = 50
V0 = replicate(nrep*(N_gen+1), -1)
W0 = replicate(nrep*(N_gen+1), -1)
dat_out = data.frame(rep = V0, gen = V0, WT = V0, M1 = V0, M2 = V0, M12 = V0)
dat_end = data.frame(rep = V0, gen = V0, WT = V0, M1 = V0, M2 = V0, M12 = V0)

tic = Sys.time()
idx = 1 ## index to account for both gen and rep
for(i in 1:nrep){
  dat_out[idx,] = c(rep = i, gen = 0, I0)
  I_t = I0
  for(t in 1:N_gen){
    idx = idx + 1 ## update index to next generation
    I_new = data.frame(WT = 0, M1 = 0, M2 = 0, M12 = 0) ## vector of next generation
    
    for(type in 1:length(I_t)){
      n_birth = rbinom(n = 1, size = as.numeric(I_t[type]), prob = B[type]/(B[type]+D_0[type])) ## births
      off_dist = rmultinom(n = 1, size = n_birth, prob = mu_matrix[[type]]) ## offspring types
      I_new = I_new + off_dist ## add new individuals
      I_new[type] = I_new[type] + n_birth ## retain birthing parents
    } ## end over types
    
    I_t = I_new 
    dat_out[idx,] = c(i, t, I_t) ## save current generation data
    
  } ## end over gen
  dat_end[nrep, ] = c(i, t, I_t)
  
} ## end over reps
toc = Sys.time()
toc-tic









