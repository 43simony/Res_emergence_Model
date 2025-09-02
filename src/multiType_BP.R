## R wrapper function for C++ simulation code
## n_reps == number of simulations to run
## parameters == a named data frame object containing all model parameters
## batch_name == naming tag for any files generated in the run
## output_mode == numeric value to modulate simulation data output. "0" returns all
#### simulation data while "1" returns only the simulated HJ probability
multiType_BP <- function(n_reps, parameters){
  
  parvec = c( format(n_reps, scientific = F),
              
              parameters$N_WT, # 0, switch indicator 0: fixed N; 1: poisson N; 2: gamma-poisson N;
              parameters$N_M1, # 1
              parameters$N_M2, # 2
              parameters$N_M12, # 3
              
              parameters$b_rate, # 4, birth rate
              parameters$d_rate, # 5, death rate
              parameters$mu_prob, # 6, mutation probability
              
              parameters$T_max, # 7, number of branching steps
              parameters$data_out, # 8, data output type -- not in use
              parameters$batchname # 9, file name tag -- not in use
  )
  strvec = format(parvec, digits = 5)
  
  setwd("~/Desktop/Repos/Res_emergence_Model/src") ## call has to be from location of .exe file or parameter read-in fails???
  
  ## Run the model
  ## The path to the bTB cpp binary file must be set correctly in the sys call below:
  nm = paste0("./multiType_BP.exe")
  r <- system2( nm, args = strvec, stdout = TRUE)
  
  ## capture model output of the simulation
  out <- read.table(text = r, header = TRUE, sep = ';', check.names = FALSE) %>% mutate_all(as.numeric)
  
  setwd("..")
  
  return( out ) 
}

# ## Generating functions
# f_WT <- function(S, b, d, mu){
#   d + b*(1-2*mu-mu^2)*S[1]^2 + mu*(1-mu)*b*S[2]*S[1] + mu*(1-mu)*b*S[3]*S[1] + b*(mu^2)*S[4]*S[1]
# }
# 
# f_M1 <- function(S, b, d, mu){
#   d + b*(1-2*mu-mu^2)*S[2]^2 + mu*(1-mu)*b*S[1]*S[2] + mu*(1-mu)*b*S[4]*S[2] + b*(mu^2)*S[3]*S[2]
# }
# 
# f_M2 <- function(S, b, d, mu){
#   d + b*(1-2*mu-mu^2)*S[3]^2 + mu*(1-mu)*b*S[4]*S[3] + mu*(1-mu)*b*S[1]*S[3] + b*(mu^2)*S[2]*S[3]
# }
# 
# f_M12 <- function(S, b, d, mu){
#   d + b*(1-2*mu-mu^2)*S[4]^2 + mu*(1-mu)*b*S[3]*S[4] + mu*(1-mu)*b*S[2]*S[4] + b*(mu^2)*S[1]*S[4]
# }
# 
# F_S <- function(S, b, d, mu){
#   return(f_WT(S, b, d, mu), f_M1(S, b, d, mu), f_M2(S, b, d, mu), f_M12(S, b, d, mu))
# }


## Single function for generating functions
## General offspring PGF generator
f_gen <- function(S, idx, b, d, mu) {
  # index ordering for each parent type, as the generating functions are all 
  ## ordered as: no mutation, site 1 mutation, site 2 mutation, double mutation
  order <- list(
    c(1, 2, 3, 4), # WT
    c(2, 1, 4, 3), # M1
    c(3, 4, 1, 2), # M2
    c(4, 3, 2, 1)  # M12
  )[[idx]]
  d_prob = d/(b+d)
  b_prob = b/(b+d)
  
  d_prob +
    b_prob * (1 - 2*mu + mu^2) * S[order[1]]^2 +
    mu * (1 - mu) * b_prob * S[order[2]] * S[order[1]] +
    mu * (1 - mu) * b_prob * S[order[3]] * S[order[1]] +
    b_prob * (mu^2) * S[order[4]] * S[order[1]]
}

## Apply to all 4 types at once
F_S <- function(S, b, d, mu) {
  sapply(1:4, function(i) f_gen(S, i, b, d, mu))
}

## Nth iterate of F_S
F_iter <- function(N, S, b, d, mu) {
  for (i in seq_len(N)) {
    S <- F_S(S, b, d, mu)
  }
  S
}

## Example usage:
S0 <- c(1, 1, 1, 1)  # e.g., starting guess for extinction probabilities
F_iter(10, S0, b = 0.9, d = 0.1, mu = 0.05)

pars = data.frame(N_WT = 10, N_M1 = 0, N_M2 = 0, N_M12 = 0,
                  b_rate = 2, d_rate = 1, mu_prob = 1e-3, 
                  T_max = 10, data_out = 0, batchname = 'test')

out <- multiType_BP(n_reps = 500, parameters = pars)
out_end <- out[out$`T` == pars$T_max,]

all(out >= 0)
sum(out$n_M1 > 0)
sum(out$n_M2 > 0)
sum(out$n_M12 > 0)
