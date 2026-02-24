% fit the DDM using simulation and comparison of CDFs (signed RT)

% you must install BADS: https://github.com/acerbilab/bads 

% generate some data
n_trials = 500; 
true_v = .8; 
true_a = 1; 
true_z = .5; 
true_t = .2; 
signed_rt = sim_ddm(n_trials, true_v, true_a, true_z, true_t);

% compute cumulative density over reasonable range
xx = linspace(-3,3,1000); 
get_cdf = @(y) arrayfun(@(g) mean(y<=g), xx);
cdf_data = get_cdf(signed_rt);

% wrap simulation, cdf, and distance from data cdf
n_sim_trials = 5e4;  % increase for accuracy > compute time
cdf_sim = @(P) get_cdf(sim_ddm(n_sim_trials,P(1),P(2),P(3),P(4)));
fn = @(P) mean((cdf_data-cdf_sim(P)).^2); 

% optimise using BADS function
lb = [.1, 0, 0, .11]; 
ub = [3, 3, 1, .8]; 
plb = [.2, .3, .35, .18];  % plausible lower/upper bounds
pub = [1., 1.5, .65, .5];

p0 = rand(size(lb)).*(pub-plb) + plb;  % start parameters
P = bads(fn, p0, lb, ub, plb, pub)

srt_fit = sim_ddm(n_sim_trials,P(1),P(2),P(3),P(4)); 
cdf_fit = get_cdf(srt_fit); 

clf
nexttile
hold on 
plot(xx, cdf_data)
plot(xx, cdf_fit)
legend({'Data','Fitted'})
xlabel('Signed RT'); ylabel('Probability of observation')
title('Cumulative distributions')

nexttile
hold on 
[y, xx] = kde(signed_rt, 'evaluationpoints', xx, 'bandwidth','plug-in');
plot(xx,y)
[y, xx] = kde(srt_fit, 'evaluationpoints', xx, 'bandwidth','plug-in');
plot(xx,y)
legend({'Data','Fitted'})
xlabel('Signed RT'); ylabel('Density')
title('Smoothed distributions')

nexttile
hold on 
bar((1:4)-.1, [true_v, true_a, true_z, true_t], .2)
bar((1:4)+.1, P, .2)
xticks(1:4)
xticklabels({'v','a','z','t'})
xlabel('Parameter')
legend({'True','Fitted'})

function [signed_rt] = sim_ddm(n_trials, v,a,z,t)
% function to simulate DDM trials (see DDM.m) 
s = 1;
dt = .001; 
time = 0:dt:25; 
signed_rt = nan(n_trials,1);

for n = 1:n_trials
    x = z*a; 
    for j = 2:numel(time)
        x = x + v*dt + s*sqrt(dt)*randn; 
        if x >= a || x <= 0
            signed_rt(n) = t + j*dt;
            if x <= 0 
                signed_rt(n) = -1*signed_rt(n); 
            end
            break
        end
    end

    if j == numel(time)
        rt(n) = inf; 
        choice(n) = -1; 
    end
end

end

