
% settings
n_trials = 500; 
sigma = .5;  % internal noise variance
true_s = 1; 

% criterion in log odds of hypotheses (s=s, s=0); 
B = 0;  % optimal criterion is 0 for equal prior and symmetric utility

% criterion in sensory measurement space
xb = true_s/2 + (sigma^2*B)/true_s;

% probability of stimulus present (prior)
p_s = .5;

x = nan(n_trials,1);
r = nan(n_trials,1);
s = nan(n_trials,1);

% generate random samples of x given s 
for n = 1:n_trials
    % sample true state, s (with prior probability)
    s(n) = randsample([0, true_s], 1, true, [1-p_s, p_s]);

    % generate internal sample, x
    x(n) = randn*sigma + s(n);

    % threshold in x 
    r(n) = x(n) >= xb;
    
    % formula for log odds with threshold B (identical result)
    logodds = true_s*x/2 - sigma^2/(2*sigma^2);
end

% compute d' 
H = sum(s==1 & r==1) ./ sum(s==1);
F = sum(s==0 & r==1) ./ sum(s==0);

dprime = norminv(H) - norminv(F);
true_d = true_s / sigma;

% compute criterion 
crit = dprime/2 - norminv(H);

fprintf("true d': %.2f, est.: %.2f\ntrue B: %.2f, est.: %.2f\n\n", ...
    true_d, dprime, B, crit)

clf
nexttile
hold on 
xx = linspace(-4*sigma, 4*sigma+true_s);
s0 = normpdf(xx, 0, sigma); 
s1 = normpdf(xx, true_s, sigma); 

plot(xx, s0)
plot(xx, s1)
xline(xb)
xlabel('x'); ylabel('Density')
legend({'Null distribution','s distribution','Optimal criterion'})
title('Signal and noise distributions')

nexttile
hold on 
bar([1,4], [true_d, B], .2)
bar([2,5], [dprime, crit], .2)
xticks([1.5, 4.5])
xticklabels({"d'", 'B'})
xlabel('Parameter')
legend({'True','Estimated'})
title('SDT parameter vs estimate')

nexttile
hold on 
jit = .02;
scatter(x(s==true_s&r==1), true_s+randn(1,sum(s==true_s&r==1))*jit)
scatter(x(s==true_s&r==0), true_s+randn(1,sum(s==true_s&r==0))*jit)
scatter(x(s==0&r==0), randn(1,sum(s==0&r==0))*jit)
scatter(x(s==0&r==1), randn(1,sum(s==0&r==1))*jit)

xline(xb)

legend({'Hits','Misses','Correct rejections','False alarms','Criterion'}, 'location','best')
xlabel('x')
ylabel('s (+ display jitter)')
title('Internal responses by choice')