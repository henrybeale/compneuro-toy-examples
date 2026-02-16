% this script shows how you can use bayes theorem to invert a generative
% probabilistic model. this allows for decoding a stimulus from
% measurements/observations. it is similar to maximum likelihood
% estimation and identical when the prior is uniform. we can use a grid
% approximation to avoid searching through the probability model
% analytically (autodiff is also another option which could be implemented
% elsewhere and is easy in python).

sdomain = linspace(0,100,100); 
p_s = ones(size(sdomain)) ./ numel(sdomain);  % uniform prior

% generative model
n_observables = 6;  % arbitrary random number of measurements
model = @(s) randn(n_observables,1)*3 + s; 

n_trials = 10; 
s = nan(n_trials,1);
X = nan(n_trials, n_observables);

for n = 1:n_trials
    % sample state/stimulus, s, with replacement
    s(n) = randsample(sdomain, 1, true, p_s);

    % generate noisy observations
    X(n,:) = model(s(n)); 
end

error('not finished yet')

clf
nexttile
hold on 
cols = colororder;
for n = 1:5
    plot(X(n,:), 1:n_observables, 'o', 'Color', cols(n,:));
    xline(s(n), 'color', cols(n,:))
end

legend({'Observations','Stimulus'})
yticks(1:n_observables)
xlabel('Stimulus');
ylabel('Observation');
title('Noisy Observations vs Stimulus');