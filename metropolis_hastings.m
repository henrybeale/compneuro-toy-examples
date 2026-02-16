
burnin = 5e2; 
max_steps = burnin+5e3; 


% proposal distribution 
g = @(x) randn*1 + x;  % Norm(0,1) 

% desired distribution
% P = @(x) normpdf(x, 20, 1);
P = @(x) .6*normpdf(x, 10, 3) + .4*normpdf(x, 20, 5) ;

X = nan(max_steps,1);
X(1) = randn; % random initialisation

for t = 2:max_steps
    % proposal
    xp = g(X(t-1)); 

    % compute product terms (use log)
    a1 = log(P(xp)) - log(P(X(t-1)));
    a2 = log(normpdf(X(t-1), xp)) - log(normpdf(xp, X(t-1))); 

    % acceptance probability
    alpha = min(1, exp(a1+a2)); 
    if rand <= alpha 
        X(t) = xp; 
    else
        X(t) = X(t-1);
    end
end


clf
nexttile
hold on 
plot(1:burnin, X(1:burnin))
plot(burnin+1:numel(X), X(burnin+1:end))
legend({'Burn-in','Samples'})
xlabel('Step'); ylabel('x')

nexttile
hold on 
[p,x] = kde(X(burnin+1:end)); 
plot(x,p)
plot(x, P(x), 'k--')
legend({'Sampled','Target'})
xlabel('s'); ylabel('p(s)')







