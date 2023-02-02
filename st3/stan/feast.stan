data {
  int N;              // Number of taxa
  array[N] int x;     // Sink counts
  int K;              // Number of known sources
  array[K, N] int y;  // Source counts

  // Priors
  real<lower=0, upper=1> unknown_mu;
  real<lower=0> unknown_kappa;
}

parameters {
  simplex[K+1] mix_prop;
  array[K+1] simplex[N] true_rel_ab;
}

transformed parameters {
  vector[N] beta_var;
  for (j in 1:N) {
    real beta_var_taxa = 0;
    for (i in 1:K+1) {
      beta_var_taxa += (mix_prop[i] * true_rel_ab[i, j]);
    }
    beta_var[j] = beta_var_taxa;
  }
}

model {
  mix_prop[K+1] ~ beta_proportion(unknown_mu, unknown_kappa);

  for (i in 1:K) {
    y[i] ~ multinomial(true_rel_ab[i]);
  }
  x ~ multinomial(beta_var);
}
