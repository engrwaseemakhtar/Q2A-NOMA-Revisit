% MATLAB code for Q-learning algorithm for sum rate optimization in Q²A-NOMA network

% Initialization
L = 10; % total number of devices (example value)
B = 20e6; % total bandwidth in Hz (example value)
P_noma = 10^(45/10) / 1000; % maximum allowable power in Watts (45dBm)
epsilon = 0.1; % exploration probability
num_iterations = 1000; % number of iterations
learning_rate = 0.1; % learning rate
discount_factor = 0.9; % discount factor

% Define state and action space
channel_conditions = rand(L, 1); % Example channel conditions
target_data_rates = rand(L, 1) * B; % Random target data rates
states = [channel_conditions, target_data_rates]; % Define state space as a combination of channel conditions and target data rates
actions = combvec(0:P_noma/10:P_noma, linspace(1, B, 10), 1:L)'; % Define action space as combinations of power levels, sub-band widths, and device counts

% Initialize Q-table
Q = zeros(size(states, 1), size(actions, 1));

% Initial state and action
state_idx = randi(size(states, 1)); % Define initial state
action_idx = randi(size(actions, 1)); % Define initial action

% Initialize reward tracking
total_rewards = zeros(num_iterations, 1);
exploration_rewards = zeros(num_iterations, 1);
exploitation_rewards = zeros(num_iterations, 1);

% Q-learning algorithm
for iter = 1:num_iterations
    total_reward = 0;
    exploration_reward = 0;
    exploitation_reward = 0;
    for sub_band = 1:L
        if rand() < epsilon
            % Exploration: select a random action
            action_idx = randi(size(actions, 1));
            exploration = true;
        else
            % Exploitation: select the action with max Q-value for the current state
            [~, action_idx] = max(Q(state_idx, :));
            exploration = false;
        end
        
        % Take action and observe new state and reward
        new_state_idx = randi(size(states, 1)); % Define state transition (example)
        reward = calculate_reward(states(new_state_idx, :), actions(action_idx, :), B, P_noma); % Define reward calculation
        
        % Update Q-value
        Q(state_idx, action_idx) = Q(state_idx, action_idx) + learning_rate * ...
            (reward + discount_factor * max(Q(new_state_idx, :)) - Q(state_idx, action_idx));
        
        % Accumulate reward
        total_reward = total_reward + reward;
        if exploration
            exploration_reward = exploration_reward + reward;
        else
            exploitation_reward = exploitation_reward + reward;
        end
        
        % Transition to the new state
        state_idx = new_state_idx;
    end
    total_rewards(iter) = total_reward;
    exploration_rewards(iter) = exploration_reward;
    exploitation_rewards(iter) = exploitation_reward;
end

% Calculate mean rewards
window_size = 100; % Moving window size for mean reward calculation
mean_total_rewards = movmean(total_rewards, window_size);
mean_exploration_rewards = movmean(exploration_rewards, window_size);
mean_exploitation_rewards = movmean(exploitation_rewards, window_size);

% Extract optimal policy from Q-table
[~, optimal_policy] = max(Q, [], 2); % Define how to extract optimal policy

% Plot only mean rewards
figure;
hold on;
plot(mean_total_rewards, 'k', 'DisplayName', 'Mean Total Reward');
plot(mean_exploration_rewards, 'm', 'DisplayName', 'Mean Exploration Reward');
plot(mean_exploitation_rewards, 'c', 'DisplayName', 'Mean Exploitation Reward');
xlabel('Iteration');
ylabel('Mean Reward');
grid on
title('Mean Reward during Q-learning iterations');
legend show;
hold off;

% Function to calculate reward
function reward = calculate_reward(state, action, B, P_noma)
    % Extract state and action components
    channel_condition = state(1);
    target_data_rate = state(2);
    power_level = action(1);
    sub_band_width = action(2);
    device_count = action(3);

    % Define parameters for IIoT network
    noise_power_density = 10^(-173/10) / 1000; % in W/Hz
    bandwidth = sub_band_width; % in Hz
    signal_power = power_level; % power assigned to the device in Watts
    
    % Generate channel gains for each device in the sub-band
    h = abs(randn(device_count, 1)); % Rayleigh channel gains (example)
    
    % Calculate SINR for each device in the sub-band
    SINR = zeros(device_count, 1);
    for l = 1:device_count
        interference_power = sum(h(1:l-1).^2 .* signal_power); % interference from previous devices
        SINR(l) = (h(l)^2 * signal_power) / (interference_power + noise_power_density * bandwidth);
    end
    
    % Calculate achievable rates for each device
    rates = bandwidth * log2(1 + SINR); % achievable rates in bits/s/Hz
    
    % Calculate sum rate and compare with target data rate
    if all(rates >= target_data_rate)
        reward = sum(rates) / target_data_rate; % total achievable sum rate divided by the target data rate
    else
        reward = 0; % no reward if any device does not meet its target data rate
    end
end
