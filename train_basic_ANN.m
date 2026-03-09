% this example shows how to use matlab to construct and train a very simple
% neural network (mostly just to show how to use the functions).

% simple function to learn:
x = linspace(-3,3,400);
% y = sin(1./x);
y = sin(2*pi*x/2) + randn(size(x))*.1;

% normalise input 
x_norm = x ./ std(x); 

% define architecture
layers = [
    featureInputLayer(1)
    fullyConnectedLayer(128)
    reluLayer

    fullyConnectedLayer(128)
    reluLayer

    fullyConnectedLayer(32)
    reluLayer

    fullyConnectedLayer(1)];

% training options
options = trainingOptions('adam',...
    'maxepochs',500,...
    'plots','none');

% train 
net = trainnet(x_norm', y', layers, 'mse', options)

% get the predictions
y_pred = predict(net, x_norm');

% view the activations in a specific layer (see names in net.Layers)
acts = minibatchpredict(net, x_norm', 'outputs', 'relu_2');

%
clf
nexttile
hold on 
plot(x,y,'.')
plot(x,y_pred)
legend({'training data','prediction'})
xlabel('input')
ylabel('output')
title('Learned (noisy) function')

nexttile
imagesc(x_norm, 1:size(acts,2), acts')
xlabel('input')
ylabel('ReLU neuron')
colorbar
title('Example layer activations')