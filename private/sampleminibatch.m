function [tt, numMiniBatches] = sampleminibatch(T,options)
% Split time series in mini-batches

% WARNING: at the moment the sampling will not produce strictly unbiased
% estimates but will be close.

switch options.minibatch.type
    case 'Uniform'
        % Array with potential starting element of the minibatch
        iiStartMiniBatch = 1 : options.minibatch.lengthMiniBatch : T;
        if mod(T, options.minibatch.lengthMiniBatch) ~= 0
            % Discard last, smaller, minibatch 
            % TODO: we should not discard data!
            iiStartMiniBatch = iiStartMiniBatch(1:end-1);
        end
        % Randomly sample subset of the time series
        tmp = randperm(numel(iiStartMiniBatch), 1);
        disp([char(8) ', Mini #' num2str(tmp)])
        tt = iiStartMiniBatch(tmp) : ...
            iiStartMiniBatch(tmp) + options.minibatch.lengthMiniBatch - 1;
        numMiniBatches = numel(iiStartMiniBatch);
    otherwise
        error('Minibatch type not recognised.')
end

end