

% activation function and derivative
sigm = @(x) 1./ (1+exp(-x)); 
dsigm = @(x) x .* (1-x); 

% training data (XOR) 
inputs = [0,0; 0,1; 1,0; 1,1];
labels = [0; 1; 1; 0];

% settings
train_steps = 1e4;
lr = .1;  % learning rate

% network size
in_size = 2; 
hidden_size = 2; 
out_size = 1;

% initialise random weights and biases
L1_w = randn(in_size, hidden_size);
L1_b = randn(1, hidden_size); 
out_w = randn(hidden_size, out_size); 
out_b = randn(1, out_size); 

% storage for history 
train_error = nan(train_steps,1);
train_w = nan(train_steps, hidden_size*in_size);
train_b = nan(train_steps, 2); 
train_out = nan(train_steps, 4);

% train
for n = 1:train_steps
    % forward pass
    L1_out = sigm(inputs * L1_w + L1_b);
    output = sigm(L1_out * out_w + out_b);

    % error backpropagation
    error = labels - output;
    d_out = error .* dsigm(output); 

    L1_err = d_out * out_w';
    d_L1 = L1_err .* dsigm(L1_out);

    % update weights and biases
    out_w = out_w + lr*L1_out'*d_out; 
    out_b = out_b + lr*sum(d_out);
    L1_w = L1_w + lr*inputs'*d_L1; 
    L1_b = L1_b + lr*sum(d_L1);

    % storage
    train_error(n) = sqrt(mean(error.^2));  % RMS
    train_w(n,:) = L1_w(:);
    train_b(n,:) = L1_b;
    train_out(n,:) = output;
end

disp(inputs)
disp(output)

clf
nexttile
hold on 
plot(train_error)
xlabel('Training step'); ylabel('RMS error')
title('Training error')

nexttile
hold on 
plot(train_out)
legend(num2str(labels))
xlabel('training step'); ylabel('Class probability')
title('Outputs')

nexttile
hold on 
plot(train_w)
xlabel('Training step')
title('Layer weights')

nexttile
hold on
plot(train_b)
title('Hidden layer biases')

